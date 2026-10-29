from functools import lru_cache


@lru_cache(maxsize=1)
def causal_length_mask():
    import cutlass.cute as cute

    @cute.jit
    def mask(batch, head, q_idx, k_idx, seqlen_info, aux_tensors, aux_scalars=None):
        return k_idx <= aux_tensors[0][0]

    return mask


def full_decode(q, k, v, positions):
    from flash_attn.cute import flash_attn_func
    output, _ = flash_attn_func(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                             mask_mod=causal_length_mask(), aux_tensors=[positions.reshape(-1)])
    return output.transpose(1,2)
