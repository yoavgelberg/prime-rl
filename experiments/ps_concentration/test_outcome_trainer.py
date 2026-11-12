import copy

import pytest
import torch

from prime_rl.orchestrator.grpo import normalized_advantages
from prime_rl.trainer.ps_backward.outcome import OutcomeConfig, OutcomeTrainer


def test_complete_delayed_grpo_and_adam_steps_match_direct_masked_objective():
    torch.manual_seed(33)
    model = torch.nn.Linear(5, 1, bias=False).double()
    reference = copy.deepcopy(model)
    config = OutcomeConfig(global_batch_size=16, learning_rate=.001, max_gradient_norm=.3)
    options = dict(lr=config.learning_rate, betas=(.9, .95), eps=1e-8, weight_decay=0)
    optimizer = torch.optim.AdamW(model.parameters(), **options)
    expected_optimizer = torch.optim.AdamW(reference.parameters(), **options)
    trainer = OutcomeTrainer(model, optimizer, config)
    for update in range(2):
        features = torch.randn(16, 7, 5, dtype=torch.float64)
        mask = torch.arange(7)[None].expand(16, -1) % 3 != 0
        rewards = torch.tensor([[0., .25, .5, -.25, -.5, 1., .6, -.3], [.2] * 8], dtype=torch.float64)
        advantages = normalized_advantages(rewards).reshape(-1)
        for index in range(16):
            nll = model(features[index]).square()[:, 0][mask[index]].sum()
            score = torch.autograd.grad(nll / config.denominator, tuple(model.parameters()))
            trainer.add(f'{update}-{index}', index // 8, dict(zip(dict(model.named_parameters()), score)),
                float(rewards.reshape(-1)[index]), policy_version=update,
                generated_tokens=int(mask[index].sum()), negative_logp_sum=float(nll.detach()))
        stats = trainer.materialize()
        assert stats['zero_advantage_groups'] == 1
        assert len(stats['advantages']) == 16
        expected_optimizer.zero_grad(set_to_none=True)
        nll = reference(features).square()[:, :, 0]
        task = ((nll * mask).sum(-1) * advantages).sum() / (config.denominator * config.global_batch_size)
        task.backward()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-17)
            actual.grad.add_(.0003)
            expected.grad.add_(.0003)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), config.max_gradient_norm)
        for group in expected_optimizer.param_groups:
            group['lr'] = config.learning_rate * (update + 1) / config.warmup_updates
        before = [p.detach().clone() for p in reference.parameters()]
        expected_optimizer.step()
        result = trainer.step()
        assert result['policy_version'] == update + 1
        assert result['gradient_norm'] == pytest.approx(float(expected_norm), rel=1e-12)
        delta = sum((p - old).square().sum() for p, old in zip(reference.parameters(), before)).sqrt()
        assert result['weight_update_norm'] == pytest.approx(float(delta.detach()), rel=1e-12)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-17)
        assert not trainer.groups and not trainer.records


def test_missing_rollouts_and_stale_versions_cannot_change_weights():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = OutcomeTrainer(model, optimizer)
    gradient = dict(weight=torch.ones_like(model.weight))
    with pytest.raises(ValueError, match='stale'):
        trainer.add('episode', 0, gradient, 1., policy_version=1, generated_tokens=4, negative_logp_sum=1.)
    trainer.add('episode', 0, gradient, 1., policy_version=0, generated_tokens=4, negative_logp_sum=1.)
    old = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match='64 fresh'):
        trainer.materialize()
    with pytest.raises(RuntimeError, match='Materialize'):
        trainer.step()
    torch.testing.assert_close(model.weight, old, rtol=0, atol=0)


def test_native_weight_refresh_preserves_storage_and_matches_every_fused_projection():
    from types import SimpleNamespace
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from prime_rl.trainer.ps_backward.qwen3 import update_inference_weights

    config = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attention_dropout=0, tie_word_embeddings=True)
    source = Qwen3ForCausalLM(config)
    def linear(a, b):
        return torch.nn.Linear(a, b, bias=False).bfloat16()
    layers = []
    for layer in source.model.layers:
        attn = SimpleNamespace(qkv_proj=linear(32, 64), o_proj=linear(32, 32),
            q_norm=torch.nn.RMSNorm(8).bfloat16(), k_norm=torch.nn.RMSNorm(8).bfloat16())
        layers.append(SimpleNamespace(self_attn=attn,
            input_layernorm=torch.nn.RMSNorm(32).bfloat16(),
            post_attention_layernorm=torch.nn.RMSNorm(32).bfloat16(),
            mlp=SimpleNamespace(gate_up_proj=linear(32, 96), down_proj=linear(48, 32))))
    embed = torch.nn.Embedding(41, 32).bfloat16()
    target = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed, norm=torch.nn.RMSNorm(32).bfloat16(),
        layers=layers), lm_head=linear(32, 41))
    target.lm_head.weight = embed.weight
    tensors = [embed.weight, target.model.norm.weight]
    for layer in layers:
        tensors += [layer.self_attn.qkv_proj.weight, layer.self_attn.o_proj.weight,
            layer.self_attn.q_norm.weight, layer.self_attn.k_norm.weight,
            layer.input_layernorm.weight, layer.post_attention_layernorm.weight,
            layer.mlp.gate_up_proj.weight, layer.mlp.down_proj.weight]
    pointers = [tensor.data_ptr() for tensor in tensors]
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(.0137)
    update_inference_weights(source, target)
    assert pointers == [tensor.data_ptr() for tensor in tensors]
    assert target.lm_head.weight is target.model.embed_tokens.weight
    torch.testing.assert_close(target.model.embed_tokens.weight, source.model.embed_tokens.weight.bfloat16())
    torch.testing.assert_close(target.model.norm.weight, source.model.norm.weight.bfloat16())
    for src, dst in zip(source.model.layers, target.model.layers, strict=True):
        torch.testing.assert_close(dst.self_attn.qkv_proj.weight, torch.cat([
            getattr(src.self_attn, name).weight.bfloat16() for name in ('q_proj', 'k_proj', 'v_proj')]))
        torch.testing.assert_close(dst.mlp.gate_up_proj.weight,
            torch.cat([src.mlp.gate_proj.weight.bfloat16(), src.mlp.up_proj.weight.bfloat16()]))
        for name in ('input_layernorm', 'post_attention_layernorm'):
            torch.testing.assert_close(getattr(dst, name).weight, getattr(src, name).weight.bfloat16())
        for name in ('o_proj', 'q_norm', 'k_norm'):
            torch.testing.assert_close(getattr(dst.self_attn, name).weight, getattr(src.self_attn, name).weight.bfloat16())
        torch.testing.assert_close(dst.mlp.down_proj.weight, src.mlp.down_proj.weight.bfloat16())
