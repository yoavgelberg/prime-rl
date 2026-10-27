import torch.nn.functional as F


def dense_ps_attention(q, k, v, q_indices, k_indices, prefix_length, window):
    """Small independent reference; never use its dense mask on long traces."""
    allowed = (
        (k_indices[None, :] >= 0)
        & (k_indices[None, :] <= q_indices[:, None])
        & ((k_indices[None, :] < prefix_length) | (k_indices[None, :] >= q_indices[:, None] - window + 1))
    )
    groups = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(groups, dim=1),
        v.repeat_interleave(groups, dim=1),
        attn_mask=allowed,
        dropout_p=0,
        scale=q.shape[-1] ** -0.5,
    )


def ps_attention(q, k, v, q_indices, k_indices, prefix_length, window, backend, plan=None):
    if window < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("PS needs a positive window and integral GQA groups")
    if backend == "dense":
        return dense_ps_attention(q, k, v, q_indices, k_indices, prefix_length, window)
    if backend == "fa4":
        from .fa4 import fa4_ps_attention

        return fa4_ps_attention(q, k, v, q_indices, k_indices, prefix_length, window, plan)
    raise ValueError(f"Unknown PS attention backend: {backend}")
