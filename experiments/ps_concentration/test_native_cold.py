import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from .linear import ProjectionCache
from .native_cold import NativeCold
from .native_teacher import NativeTeacher


def test_original_position_cold_cache_matches_recomputation_through_multiple_evictions(monkeypatch):
    torch.manual_seed(91)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=51, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=1024, attention_dropout=0.)).float().eval()
    monkeypatch.setattr(ProjectionCache, 'linear', lambda self, inputs, weight, bias=None:
        F.linear(inputs.bfloat16(), weight.bfloat16(), None if bias is None else bias.bfloat16()).float())
    tokens = torch.randint(0, 51, (20,)).tolist()
    positions = list(range(4)) + list(range(400, 416))
    stream = NativeCold(base, prefix=4, window=16, copies=2, backend='dense')
    with torch.no_grad():
        logits = stream.prefill(tokens, positions)
        pinned = [key[:, :, :4].clone() for key in stream.keys]
        for index in range(35):
            reference = NativeTeacher(base, 16, backend='dense')
            ids, absolute = torch.tensor(tokens), torch.tensor(positions)
            hidden = reference.segment(ids, absolute, absolute, 4)
            expected = reference.linear(hidden[-1:], base.lm_head.weight)
            torch.testing.assert_close(logits, expected.expand_as(logits), rtol=.01, atol=.01)
            token = int(logits[0].argmax())
            tokens.append(token); positions.append(positions[-1] + 1)
            logits = stream.step(torch.tensor([token, token]))
        assert stream.next_position == 451
        assert stream.indices.shape == (20,)
        assert all(value.shape[2] == 20 for value in [*stream.keys, *stream.values])
        for actual, original in zip(stream.keys, pinned, strict=True):
            torch.testing.assert_close(actual[:, :, :4], original, rtol=0, atol=0)


def test_independent_branches_do_not_share_mutable_kv(monkeypatch):
    torch.manual_seed(4)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=51, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=256, attention_dropout=0.)).float().eval()
    monkeypatch.setattr(ProjectionCache, 'linear', lambda self, inputs, weight, bias=None:
        F.linear(inputs.bfloat16(), weight.bfloat16()).float())
    stream = NativeCold(base, 4, 16, copies=2, backend='dense')
    stream.prefill(list(range(20)), list(range(4)) + list(range(100, 116)))
    stream.step(torch.tensor([3, 37]))
    assert any(not torch.equal(keys[0, :, 4], keys[1, :, 4]) for keys in stream.keys)
    assert torch.equal(stream.keys[0][0, :, :4], stream.keys[0][1, :, :4])
