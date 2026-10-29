"""Append already queued tokens without rebuilding retained hidden states."""
import torch

from prime_rl.trainer.ps_backward.fa4 import fa4_ps_attention, sparse_ps_plan
from prime_rl.trainer.ps_backward.model import rotate_half


@torch.inference_mode()
@torch.autocast('cuda', dtype=torch.bfloat16)
def append_tokens(stream, tokens):
    n, count = stream.consumed, tokens.shape[1]
    assert stream.position == n and count > 0
    queries = torch.arange(n, n+count, device=tokens.device)
    if stream.attention == 'ps':
        assert n >= stream.capacity and count <= stream.window
        slots = torch.arange(stream.window, device=tokens.device)
        oldest_slot = (n-stream.prefix) % stream.window
        history = torch.cat((torch.arange(stream.prefix, device=tokens.device),
                             n-stream.window+(slots-oldest_slot) % stream.window))
        indices = torch.cat((history, queries))
        write_slots = stream.prefix+(queries-stream.prefix) % stream.window
        window = stream.window
    else:
        assert n+count <= stream.capacity
        used = min(stream.capacity, ((n+count+2047)//2048)*2048)
        indices = torch.arange(used, device=tokens.device)
        write_slots = queries
        window = stream.capacity+1
    major = torch.cuda.get_device_capability(tokens.device)[0]
    plan = sparse_ps_plan(queries, indices, stream.prefix, window,
                          (128 if major == 9 else 256,128))
    hidden = stream.base.model.embed_tokens(tokens)
    cos, sin = stream.base.model.rotary_emb(hidden, queries[None].expand(len(tokens),-1))
    cos, sin = cos[:,None], sin[:,None]
    for layer, keys, values in zip(stream.base.model.layers,stream.keys,stream.values,strict=True):
        normalized = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q,k,v = [stream.projections.project(proj,normalized).view(len(tokens),count,-1,attn.head_dim).transpose(1,2)
                 for proj in (attn.q_proj,attn.k_proj,attn.v_proj)]
        if hasattr(attn,'q_norm'):
            q,k = attn.q_norm(q),attn.k_norm(k)
        q,k = q*cos+rotate_half(q)*sin,k*cos+rotate_half(k)*sin
        q,k,v = q.bfloat16(),k.bfloat16(),v.bfloat16()
        if stream.attention == 'ps':
            attended_keys = torch.cat((keys,k),dim=2)
            attended_values = torch.cat((values,v),dim=2)
        else:
            keys.index_copy_(2,write_slots,k)
            values.index_copy_(2,write_slots,v)
            attended_keys,attended_values = keys[:,:,:used],values[:,:,:used]
        output = fa4_ps_attention(q,attended_keys,attended_values,queries,indices,
                                  stream.prefix,window,plan)
        if stream.attention == 'ps':
            keys.index_copy_(2,write_slots,k)
            values.index_copy_(2,write_slots,v)
        hidden = hidden+stream.projections.project(attn.o_proj,output.transpose(1,2).reshape(len(tokens),count,-1))
        x = layer.post_attention_layernorm(hidden)
        gated = layer.mlp.act_fn(stream.projections.project(layer.mlp.gate_proj,x))*stream.projections.project(layer.mlp.up_proj,x)
        hidden = hidden+stream.projections.project(layer.mlp.down_proj,gated)
    stream.position += count
    stream.consumed += count
