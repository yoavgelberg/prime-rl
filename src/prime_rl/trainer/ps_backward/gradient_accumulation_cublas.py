"""cuBLAS GEMMs with device-selected trajectory accumulators, including graph replay."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _operand_pointers(X, DY, POINTERS, OFFSET: tl.constexpr):
    tl.store(POINTERS, X.to(tl.int64))
    tl.store(POINTERS + 1, (DY + OFFSET).to(tl.int64))


def _check(status):
    if status:
        raise RuntimeError(f"Direct gradient cuBLAS operation failed with status {status}")


@lru_cache(maxsize=1)
def _library():
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/libcublas.so" in line}
    packaged = {path for path in paths if Path(torch.__file__).resolve().parents[1] in Path(path).parents}
    paths = packaged or paths
    if len(paths) != 1:
        raise RuntimeError(f"Expected one loaded cuBLAS library, found {paths}")
    library = ctypes.CDLL(paths.pop())
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    library.cublasCreate_v2.argtypes = [ctypes.POINTER(pointer)]
    library.cublasSetStream_v2.argtypes = [pointer, pointer]
    library.cublasSetMathMode.argtypes = [pointer, integer]
    library.cublasSetWorkspace.argtypes = [pointer, pointer, ctypes.c_size_t]
    library.cublasGemmBatchedEx.argtypes = [
        pointer,
        integer,
        integer,
        integer,
        integer,
        integer,
        pointer,
        pointer,
        integer,
        integer,
        pointer,
        integer,
        integer,
        pointer,
        pointer,
        integer,
        integer,
        integer,
        integer,
        integer,
    ]
    for name in (
        "cublasCreate_v2",
        "cublasSetStream_v2",
        "cublasSetMathMode",
        "cublasSetWorkspace",
        "cublasGemmBatchedEx",
    ):
        getattr(library, name).restype = integer
    return library


@lru_cache(maxsize=None)
def _handle(device, stream):
    library = _library()
    handle = ctypes.c_void_p()
    _check(library.cublasCreate_v2(ctypes.byref(handle)))
    _check(library.cublasSetStream_v2(handle, stream))
    # CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION.
    _check(library.cublasSetMathMode(handle, 16))
    workspace = torch.empty(32 * 1024**2, device=device, dtype=torch.uint8)
    _check(library.cublasSetWorkspace(handle, workspace.data_ptr(), workspace.numel()))
    return handle, workspace


def accumulate(dy, x, pointers, indices, widths):
    if x.dtype not in (torch.bfloat16, torch.float16) or dy.dtype != x.dtype:
        raise ValueError("Direct cuBLAS gradients require BF16 or FP16 operands")
    if x.stride(1) != 1 or dy.stride(1) != 1:
        raise ValueError("Direct cuBLAS gradients require contiguous operand columns")
    library = _library()
    handle, _ = _handle(x.device, torch.cuda.current_stream(x.device).cuda_stream)
    operands = torch.empty(2, device=x.device, dtype=torch.int64)
    one = ctypes.c_float(1)
    dtype = 14 if x.dtype == torch.bfloat16 else 2  # CUDA_R_16BF / CUDA_R_16F.
    offset = 0
    for index, width in zip(indices, widths, strict=True):
        _operand_pointers[(1,)](x, dy, operands, offset)
        # Row-major dW = dY.T @ X is column-major dW.T = X.T @ dY.
        _check(
            library.cublasGemmBatchedEx(
                handle,
                0,
                1,
                x.shape[1],
                width,
                x.shape[0],
                ctypes.byref(one),
                operands.data_ptr(),
                dtype,
                x.stride(0),
                operands.data_ptr() + 8,
                dtype,
                dy.stride(0),
                ctypes.byref(one),
                pointers.data_ptr() + 8 * index,
                0,
                x.shape[1],
                1,
                68,
                -1,
            )
        )
        offset += width
