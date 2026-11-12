import json

import pytest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from prime_rl.trainer.ps_backward.checkpoint import save_snapshot
from prime_rl.trainer.ps_backward.model import QwenPS
from prime_rl.trainer.ps_backward.outcome import OutcomeConfig
from experiments.ps_concentration.export_checkpoint import export_snapshot


def test_deferred_export_preserves_effective_weights_token_stops_and_training_identity(tmp_path):
    torch.manual_seed(91)
    config = Qwen3Config(vocab_size=41, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attention_dropout=0, tie_word_embeddings=True)
    reference = tmp_path / 'fixture-revision'
    config.save_pretrained(reference)
    GenerationConfig(eos_token_id=[3, 4], pad_token_id=0).save_pretrained(reference)
    model = Qwen3ForCausalLM(config).float()
    adapter = QwenPS(model, 16, backend='dense')
    snapshot = tmp_path / 'checkpoints/latest.pt'
    metadata = dict(model_revision=reference.name, reference_checkpoint=str(reference),
        attention='ps', window=1024, evaluation_deferred=True, best_update=None)
    save_snapshot(snapshot, adapter, {}, policy_version=7, outcome_config=OutcomeConfig(),
        rngs=[], metadata=metadata)
    snapshot_bytes = snapshot.read_bytes()
    class Tokenizer:
        def save_pretrained(self, path):
            (path / 'tokenizer-fixture.json').write_text('{}')
    result = export_snapshot(snapshot, tokenizer=Tokenizer())
    output = snapshot.parent / 'policy-0007'
    restored = AutoModelForCausalLM.from_pretrained(output, torch_dtype=torch.bfloat16, attn_implementation='eager')
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value.bfloat16(), rtol=0, atol=0)
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    assert restored.generation_config.eos_token_id == [3, 4]
    assert snapshot.read_bytes() == snapshot_bytes
    receipt = json.loads((output / 'experiment.json').read_text())
    assert receipt['policy_version'] == result['policy_version'] == 7
    assert receipt['evaluation_deferred'] is True and receipt['best_update'] is None
    assert 'no validation selection' in receipt['checkpoint_selection']
    with pytest.raises(FileExistsError, match='existing export'):
        export_snapshot(snapshot, tokenizer=Tokenizer())
    with pytest.raises(ValueError, match='immutable model revision'):
        export_snapshot(snapshot, tmp_path / 'other', reference=tmp_path / 'different-revision', tokenizer=Tokenizer())


def test_export_rejects_a_snapshot_without_a_trained_update(tmp_path):
    state = dict(format='prime-rl-recorded-grpo-v1', outcome_config=OutcomeConfig().__dict__, policy_version=0)
    snapshot = tmp_path / 'latest.pt'
    torch.save(state, snapshot)
    with pytest.raises(ValueError, match='No trained optimizer update'):
        export_snapshot(snapshot)
