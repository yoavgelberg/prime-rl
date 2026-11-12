import json

import pytest
import torch

from prime_rl.trainer.ps_backward.checkpoint import restore_snapshot, save_inference, save_snapshot
from prime_rl.trainer.ps_backward.outcome import OutcomeConfig


def test_snapshot_restores_exact_next_adam_update_and_rng(tmp_path):
    torch.manual_seed(11)
    model = torch.nn.Linear(5, 3).double()
    config = OutcomeConfig()
    options = dict(lr=5e-6, betas=(.9, .95), eps=1e-8, weight_decay=0)
    optimizer = torch.optim.AdamW(model.parameters(), **options)
    def step(model, optimizer):
        value = torch.randn(7, 5, dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        model(value).square().sum().backward()
        optimizer.step()
        return value
    step(model, optimizer)
    path = tmp_path / 'latest.pt'
    (tmp_path / 'latest.pt.tmp-1').write_text('interrupted attempt')
    rng = torch.get_rng_state()
    metadata = dict(reference_revision='immutable-reference', sampler='unfiltered-.6', training_seed=11)
    save_snapshot(path, model, optimizer.state_dict(), policy_version=1,
        outcome_config=config, rngs=[dict(cpu=rng)], metadata=metadata)
    expected_input = step(model, optimizer)
    restored = torch.nn.Linear(5, 3).double()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), **options)
    state = restore_snapshot(path, restored, restored_optimizer, outcome_config=config,
                             expected_metadata=metadata)
    torch.set_rng_state(state['rngs'][0]['cpu'])
    actual_input = step(restored, restored_optimizer)
    torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
    for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(restored_optimizer.state.values(), optimizer.state.values(), strict=True):
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert json.loads(path.with_suffix('.pt.json').read_text())['policy_version'] == 1
    assert len(list(tmp_path.glob('latest.pt.tmp-1.interrupted-*'))) == 1
    with pytest.raises(ValueError, match='identity changed'):
        restore_snapshot(path, restored, restored_optimizer, outcome_config=config,
                         expected_metadata=dict(reference_revision='different-reference'))
    with pytest.raises(ValueError, match='configuration changed'):
        restore_snapshot(path, restored, restored_optimizer,
                         outcome_config=OutcomeConfig(learning_rate=1e-5), expected_metadata=metadata)


def test_inference_checkpoint_reloads_all_native_bf16_weights_and_ties(tmp_path):
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attention_dropout=0, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(config)
    class Tokenizer:
        def save_pretrained(self, path):
            (path / 'tokenizer-fixture.json').write_text('{}')
    dtype_attribute = "dtype" if hasattr(model.config, "dtype") else "torch_dtype"
    original_dtype = getattr(model.config, dtype_attribute)
    path = tmp_path / 'checkpoint-0001'
    save_inference(path, model, Tokenizer(), policy_version=1, metadata=dict(task='native-memory-game'))
    assert getattr(model.config, dtype_attribute) == original_dtype
    restored = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='eager')
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value.bfloat16(), rtol=0, atol=0)
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    assert json.loads((path / 'experiment.json').read_text())['policy_version'] == 1


def test_sharded_fused_optimizer_restores_step_counters_to_parameter_device():
    from types import SimpleNamespace
    from prime_rl.trainer.ps_backward.checkpoint import restore_optimizer_step_devices
    parameter=torch.nn.Parameter(torch.empty(2,device='meta'))
    original=torch.tensor(3.)
    local=SimpleNamespace(param_groups=[dict(params=[parameter],fused=True)],
        state={parameter:dict(step=original)})
    restore_optimizer_step_devices(SimpleNamespace(optim=local))
    assert local.state[parameter]['step'].device==parameter.device
    assert original.item()==3.
    local.param_groups[0]['fused']=False
    local.state[parameter]['step']=original
    restore_optimizer_step_devices(local)
    assert local.state[parameter]['step'] is original
