"""Convert an unsharded Qwen3 inference policy to the recorded-backward adapter."""

import torch

from .model import QwenPS


def runner_model(runner):
    return runner.model.runnable if hasattr(runner.model, "runnable") else runner.model


@torch.no_grad()
def copy_qwen3_weights(source, target):
    """Copy the current unsharded inference policy, including fused QKV/gate-up weights."""

    def copy_linear(src, dst):
        dst.weight.copy_(src.weight)
        if (src.bias is None) != (dst.bias is None):
            raise ValueError("Inference and backward projection biases differ")
        if dst.bias is not None:
            dst.bias.copy_(src.bias)

    target.model.embed_tokens.weight.copy_(source.model.embed_tokens.weight)
    target.model.norm.weight.copy_(source.model.norm.weight)
    target.lm_head.weight.copy_(source.lm_head.weight)
    for src, dst in zip(source.model.layers, target.model.layers, strict=True):
        for name in ("input_layernorm", "post_attention_layernorm"):
            getattr(dst, name).weight.copy_(getattr(src, name).weight)
        sa, da = src.self_attn, dst.self_attn
        sizes = sa.q_size, sa.kv_size, sa.kv_size
        for name, value in zip(("q_proj", "k_proj", "v_proj"), sa.qkv_proj.weight.split(sizes), strict=True):
            getattr(da, name).weight.copy_(value)
        if sa.qkv_proj.bias is not None:
            for name, value in zip(("q_proj", "k_proj", "v_proj"), sa.qkv_proj.bias.split(sizes), strict=True):
                getattr(da, name).bias.copy_(value)
        copy_linear(sa.o_proj, da.o_proj)
        da.q_norm.weight.copy_(sa.q_norm.weight)
        da.k_norm.weight.copy_(sa.k_norm.weight)
        gate, up = src.mlp.gate_up_proj.weight.chunk(2)
        dst.mlp.gate_proj.weight.copy_(gate)
        dst.mlp.up_proj.weight.copy_(up)
        copy_linear(src.mlp.down_proj, dst.mlp.down_proj)
    # to_empty does not initialize nonparameter buffers such as RoPE frequencies.
    target.model.rotary_emb = type(target.model.rotary_emb)(
        target.config, device=target.model.embed_tokens.weight.device
    )


def load_backward_model(runner, window):
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(runner.vllm_config.model_config.hf_config, torch_dtype=torch.float32)
    base.to_empty(device=torch.device("cuda", torch.cuda.current_device()))
    base.tie_weights()
    copy_qwen3_weights(runner_model(runner), base)
    return QwenPS(base, window, fp32_linear_gradients=True)
