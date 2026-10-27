"""Read FA4's Hopper WGMMA accumulator fragments without rounding to BF16."""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.interface import make_fake_bwd_tensors


class HopperPostprocess:
    def __init__(self, dim, tile_m, threads, atom_layout, swap_ab, alignment):
        self.tile_m, self.tile_d = (
            tile_m,
            ((dim + alignment - 1) // alignment) * alignment,
        )
        self.threads, self.swap_ab = threads, swap_ab
        self.layout = FlashAttentionBackwardPostprocess(
            cutlass.BFloat16,
            dim,
            90,
            tile_m,
            threads,
            atom_layout,
            swap_ab,
            hdim_multiple_of=alignment,
        )

    @cute.jit
    def __call__(self, accum: cute.Tensor, output: cute.Tensor, scale: cutlass.Float32, stream: cuda.CUstream = None):
        mma = self.layout._get_tiled_mma()
        self.kernel(accum, output, scale, mma).launch(
            grid=(
                cute.ceil_div(output.shape[1], self.tile_m),
                output.shape[2],
                output.shape[0],
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        accum: cute.Tensor,
        output: cute.Tensor,
        scale: cutlass.Float32,
        mma: cute.TiledMma,
    ):
        thread, _, _ = cute.arch.thread_idx()
        tile, head, batch = cute.arch.block_idx()
        shape = (self.tile_m, self.tile_d)
        if cutlass.const_expr(self.swap_ab):
            shape = shape[::-1]
        coordinates = mma.get_slice(thread).partition_C(
            cute.make_identity_tensor(shape)
        )
        warp_group_size = self.tile_m * self.tile_d // (self.threads // 128)
        base = tile * self.tile_m * self.tile_d + thread // 128 * warp_group_size
        for index in cutlass.range(cute.size(coordinates), unroll_full=True):
            row, col = coordinates[index]
            if cutlass.const_expr(self.swap_ab):
                row, col = col, row
            row += tile * self.tile_m
            if row < output.shape[1] and col < output.shape[3]:
                offset = base + thread % 128 * 4 + index % 4 + index // 4 * 512
                output[batch, row, head, col] = accum[batch, head, offset] * scale


@lru_cache(maxsize=None)
def _compile(dim, tile_m, threads, atom_layout, swap_ab, alignment):
    tensors = make_fake_bwd_tensors(
        cutlass.Float32, has_gqa=True, varlen_q=False, varlen_k=False
    )
    return cute.compile(
        HopperPostprocess(dim, tile_m, threads, atom_layout, swap_ab, alignment),
        tensors[11],
        tensors[5],
        cutlass.Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def convert_hopper(
    accum,
    output,
    scale,
    dim,
    tile_m,
    threads,
    atom_layout,
    swap_ab,
    alignment,
    *,
    fake_mode,
):
    kernel = _compile(dim, tile_m, threads, atom_layout, swap_ab, alignment)
    if not fake_mode:
        kernel(accum, output, scale)
