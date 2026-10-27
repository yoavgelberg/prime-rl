import copy
from types import SimpleNamespace

import torch
from helpers import tiny_model

from prime_rl.trainer.ps_backward.qwen3 import copy_qwen3_weights


def test_copy_current_fused_inference_weights_including_tied_embeddings():
    model = tiny_model(dtype=torch.float32).base
    expected_rope = model.model.rotary_emb.inv_freq.clone()
    fused = copy.deepcopy(model).to(torch.bfloat16)
    for layer in fused.model.layers:
        attn = layer.self_attn
        attn.q_size, attn.kv_size = attn.q_proj.weight.shape[0], attn.k_proj.weight.shape[0]
        attn.qkv_proj = SimpleNamespace(
            weight=torch.cat([p.weight for p in (attn.q_proj, attn.k_proj, attn.v_proj)]), bias=None
        )
        layer.mlp.gate_up_proj = SimpleNamespace(
            weight=torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight))
        )
    expected = {name: value.detach().float().clone() for name, value in fused.named_parameters()}
    with torch.device("meta"):
        model = type(model)(copy.deepcopy(model.config))
    model.to_empty(device="cpu")
    model.tie_weights()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for buffer in model.buffers():
            buffer.fill_(float("nan"))
    copy_qwen3_weights(fused, model)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    assert model.model.embed_tokens.weight is model.lm_head.weight

    torch.testing.assert_close(model.model.rotary_emb.inv_freq, expected_rope, rtol=0, atol=0)
    assert not model.model.rotary_emb.original_inv_freq.is_meta
