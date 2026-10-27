"""Upstream FA4 with an exact PS mask and sparse forward/backward traversal."""

from dataclasses import dataclass, replace
from functools import lru_cache

import torch


@dataclass(frozen=True)
class SparsePSPlan:
    forward: tuple
    backward: tuple
    block_size: tuple[int, int]
    q_indices: torch.Tensor
    k_indices: torch.Tensor
    source_indices: torch.Tensor | None = None
    coordinate_gap: int | None = None
    q_padding: int = 0
    q_tail_padding: int = 0


def _blocks(indices, block, prefix_length):
    infinity = torch.iinfo(torch.int64).max
    n = indices.numel()
    padded = torch.cat((indices, indices.new_full(((-n) % block,), -1))).reshape(-1, block)
    actual = torch.arange(padded.numel(), device=indices.device).reshape_as(padded) < n
    valid = actual & (padded >= 0)
    low = torch.where(valid, padded, infinity).amin(1)
    high = torch.where(valid, padded, -infinity).amax(1)
    prefix = valid & (padded < prefix_length)
    local = valid & (padded >= prefix_length)
    prefix_low = torch.where(prefix, padded, infinity).amin(1)
    local_low = torch.where(local, padded, infinity).amin(1)
    local_high = torch.where(local, padded, -infinity).amax(1)
    return low, high, prefix_low, local_low, local_high, ((~actual) | valid).all(1)


def _pack(allowed):
    columns = torch.arange(allowed.shape[1], device=allowed.device).expand_as(allowed)
    indices = torch.where(allowed, columns, allowed.shape[1]).sort(dim=1).values
    indices = torch.where(indices < allowed.shape[1], indices, 0).to(torch.int32)
    counts = allowed.sum(1, dtype=torch.int32)[None, None].clone(memory_format=torch.contiguous_format)
    indices = indices[None, None].clone(memory_format=torch.contiguous_format)
    return counts, indices


def sparse_ps_plan(q_indices, k_indices, prefix_length, window, block_size=(128, 128), *, source_indices=None):
    """Conservative occupied tiles; full tiles are proven valid for every token pair.

    Metadata uses a block grid, not a token-by-token attention matrix. Position
    gaps and negative padded key positions remain explicit in the exact mask.
    """
    if not q_indices.numel() or not k_indices.numel() or window < 1:
        raise ValueError("PS requires nonempty positions and a positive window")
    qlo, qhi, _, _, _, qvalid = _blocks(q_indices, block_size[0], prefix_length)
    klo, khi, prefix_lo, local_lo, local_hi, kvalid = _blocks(k_indices, block_size[1], prefix_length)
    occupied = (qhi[:, None] >= klo[None, :]) & (
        (prefix_lo[None, :] <= qhi[:, None])
        | ((local_lo[None, :] <= qhi[:, None]) & (local_hi[None, :] >= qlo[:, None] - window + 1))
    )
    full = (
        qvalid[:, None]
        & kvalid[None, :]
        & (khi[None, :] <= qlo[:, None])
        & (local_lo[None, :] >= qhi[:, None] - window + 1)
        & occupied
    )
    partial = occupied & ~full
    forward = (*_pack(partial), *_pack(full))
    backward = (*_pack(partial.T), *_pack(full.T))
    return SparsePSPlan(forward, backward, block_size, q_indices.contiguous(), k_indices.contiguous(), source_indices)


def aligned_chunk_plan(prefix_length, query_start, query_end, history_start, window, device, block_size=(256, 128)):
    """Pad unused rows/keys so live entries keep their full-sequence tile offsets."""
    q_padding = query_start % block_size[0]
    q_tail_padding = (-query_end) % block_size[0]
    queries = torch.arange(query_start - q_padding, query_end + q_tail_padding, device=device)
    prefix = torch.arange(prefix_length, device=device)
    local = torch.arange(history_start, query_end, device=device)
    padding = ((-prefix_length) % block_size[1] + history_start % block_size[1]) if history_start > prefix_length else 0
    keys = torch.cat((prefix, prefix.new_full((padding,), -1), local))
    source = None
    if padding:
        source = torch.cat(
            (
                prefix,
                prefix.new_zeros(padding),
                torch.arange(prefix_length, prefix_length + local.numel(), device=device),
            )
        )
    return replace(
        sparse_ps_plan(queries, keys, prefix_length, window, block_size, source_indices=source),
        q_padding=q_padding,
        q_tail_padding=q_tail_padding,
    )


class AlignedChunkPlans:
    """Reuse the stationary sparse tile layout; only absolute local positions move."""

    def __init__(self):
        self.templates = {}

    def get(self, prefix, start, end, history, window, device, block_size):
        padded_start = start - start % block_size[0]
        key = (
            prefix,
            window,
            str(device),
            block_size,
            start % block_size[0],
            end - start,
            start - history,
            history % block_size[1],
            history == prefix,
            start if padded_start < prefix else None,
        )
        if key not in self.templates:
            plan = aligned_chunk_plan(prefix, start, end, history, window, device, block_size)
            self.templates[key] = start, history, plan
            return plan
        original_start, original_history, plan = self.templates[key]
        shift = start - original_start
        if shift == 0:
            return plan
        queries = plan.q_indices + shift
        keys = torch.where(plan.k_indices >= original_history, plan.k_indices + shift, plan.k_indices)
        return replace(plan, q_indices=queries, k_indices=keys)


class _ReplicatePrefix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, source_indices):
        ctx.save_for_backward(source_indices)
        ctx.original_shape = tensor.shape
        return tensor.index_select(2, source_indices)

    @staticmethod
    def backward(ctx, grad):
        (indices,) = ctx.saved_tensors
        dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
        result = torch.zeros(ctx.original_shape, device=grad.device, dtype=dtype)
        result.index_add_(2, indices, grad.to(dtype))
        return result.to(grad.dtype), None


def compile_cache_sizes():
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    return tuple(len(fn.compile_cache.cache) for fn in (_flash_attn_fwd, _flash_attn_bwd))


@lru_cache(maxsize=64)
def _ps_mask(prefix_length, window, position_mode="indexed"):
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def load_position(aux, index):
        scalar_index = cute.make_rmem_tensor(1, cutlass.Int32)
        scalar_index.store(index)
        value = cute.make_rmem_tensor(1, cutlass.Int64)
        value[0] = aux[cutlass.min(scalar_index[0], aux.shape[0] - 1)]
        return value.load()

    @cute.jit
    def mask(batch, head, q_idx, k_idx, seqlen_info, aux_tensors, aux_scalars=None):
        if cutlass.const_expr(position_mode != "indexed"):
            qpos, kpos = q_idx, k_idx
        else:
            qpos = load_position(aux_tensors[0], q_idx)
            kpos = load_position(aux_tensors[1], k_idx)
        allowed = (
            (q_idx < aux_tensors[0].shape[0])
            & (k_idx < aux_tensors[1].shape[0])
            & (kpos >= 0)
            & (kpos <= qpos)
            & ((kpos < prefix_length) | (kpos >= qpos - window + 1))
        )
        return allowed

    return mask


def fa4_ps_attention(q, k, v, q_indices, k_indices, prefix_length, window, plan=None):
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FA4 PS requires CUDA FP16/BF16 tensors; no fallback is allowed")
    from flash_attn.cute import flash_attn_func
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    major = torch.cuda.get_device_capability(q.device)[0]
    if major not in (9, 10, 11):
        raise ValueError("This FA4 PS integration targets Hopper and Blackwell")
    if plan is None:
        plan = sparse_ps_plan(q_indices, k_indices, prefix_length, window, (128 if major == 9 else 256, 128))
    query_tokens = q.shape[2]
    if plan.q_padding or plan.q_tail_padding:
        q = torch.nn.functional.pad(q, (0, 0, plan.q_padding, plan.q_tail_padding))
    if plan.source_indices is not None:
        k = _ReplicatePrefix.apply(k, plan.source_indices)
        v = _ReplicatePrefix.apply(v, plan.source_indices)
    forward = BlockSparseTensorsTorch(*plan.forward, block_size=plan.block_size)
    backward = BlockSparseTensorsTorch(*plan.backward, block_size=plan.block_size)
    position_mode = "indexed"
    if plan.coordinate_gap is not None:
        if plan.coordinate_gap < 0:
            raise ValueError("A prefix-plus-interval coordinate gap must be nonnegative")
        # Translating the entire nonprefix interval preserves every PS edge.
        # RoPE still uses original positions; the attention mask needs no gap.
        position_mode = "contiguous"
    output, _ = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=q.shape[-1] ** -0.5,
        mask_mod=_ps_mask(prefix_length, window, position_mode),
        aux_tensors=[plan.q_indices, plan.k_indices],
        block_sparse_tensors=forward,
        block_sparse_tensors_bwd=backward,
    )
    return output[:, plan.q_padding : plan.q_padding + query_tokens].transpose(1, 2)
