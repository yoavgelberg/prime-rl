from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("ring", [False, True])
def test_fused_slot_metadata_clears_padding_and_handles_wraps(ring):
    from prime_rl.inference.vllm.ps_capture import _assign_rows

    starts = torch.tensor([0, 2, 3, 3], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([7, 10, 0], device="cuda", dtype=torch.int32)
    requests = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32)
    slots = torch.full((13,), 99, device="cuda", dtype=torch.int64)
    heads = torch.full((3,), 99, device="cuda", dtype=torch.int64)
    overflow = torch.zeros((), device="cuda", dtype=torch.int32)

    def assign():
        _assign_rows[(3,)](starts, lengths, requests, slots, heads, overflow, 16, 2, ring, 3, 13, 16)

    assign()
    assert slots.tolist() == [5, 6, 25] + [-1] * 10
    assert heads.tolist() == [6, 25, -1]
    assert overflow.item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assign()
    starts.copy_(torch.tensor([0, 1, 1, 1], device="cuda", dtype=torch.int32))
    lengths[0] = 18
    graph.replay()
    assert slots.tolist() == ([1] if ring else [-1]) + [-1] * 12
    assert heads.tolist() == ([1] if ring else [-1]) + [-1] * 2
    assert overflow.item() == (0 if ring else 1)


def test_fused_recording_preserves_values_and_captures_all_layer_fields(monkeypatch):
    compact = True
    from prime_rl.inference.vllm import ps_capture
    from prime_rl.inference.vllm.ps_capture import CaptureStore
    from prime_rl.inference.vllm.ps_capture_kernels import fused_norm_record, fused_silu_record
    from prime_rl.trainer.ps_backward.recorded_values import restore_values

    config = SimpleNamespace(
        additional_config={"ps_streaming": {"max_requests": 2, "ring_tokens": 16, "chunk_tokens": 8}},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=32,
                intermediate_size=48,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
            )
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=4),
    )
    tape = CaptureStore(config, "cuda")
    tape.requests["r"] = 0
    tape.token_slots[:3] = torch.tensor([2, 8, -1], device="cuda")
    for value in (*tape.layers, *tape.norm_inputs, *tape.lse, tape.final):
        value.zero_()
    torch.manual_seed(202)
    x, residual = (torch.randn(3, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    weight = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    big = torch.randn(3, 153, device="cuda", dtype=torch.bfloat16)
    qkv, q, k, attended = big[:, :64], big[:, 64:96], big[:, 96:112], big[:, 112:144]
    gate_up = torch.randn(3, 96, device="cuda", dtype=torch.bfloat16)
    lse = torch.randn(3, 4, device="cuda").T
    tape.pending_lse[0] = lse
    monkeypatch.setattr(ps_capture, "_store", None)
    expected_norm, expected_residual = fused_norm_record(0, 1, x, residual, weight, 1e-6, qkv, q, k, attended)
    expected_gated = fused_silu_record(0, gate_up)
    raw = x.float() + residual.float()
    expected = (raw * (raw.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight.float()).bfloat16()
    torch.testing.assert_close(expected_norm, expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        expected_gated,
        (torch.nn.functional.silu(gate_up[:, :48].float()) * gate_up[:, 48:].float()).bfloat16(),
        atol=1e-2,
        rtol=1e-2,
    )
    monkeypatch.setattr(ps_capture, "_store", tape)
    fused_norm_record(0, 0, x, None, weight, 1e-6)
    norm, residual_out = fused_norm_record(0, 1, x, residual, weight, 1e-6, qkv, q, k, attended)
    gated = fused_silu_record(0, gate_up)
    final, _ = fused_norm_record(0, 2, x, residual, weight, 1e-6)
    torch.testing.assert_close(norm, expected_norm, atol=0, rtol=0)
    torch.testing.assert_close(residual_out, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(gated, expected_gated, atol=0, rtol=0)
    torch.testing.assert_close(final, expected_norm, atol=0, rtol=0)
    for token, slot in enumerate((2, 8)):
        rec = restore_values(tape.layer("r", 0, slot, slot + 1))
        for field, expected in (
            ("input", x),
            ("qkv", qkv),
            ("query", q),
            ("key", k),
            ("attention", attended),
            ("norm2", norm),
            ("mid", residual_out),
            ("gate_up", gate_up),
            ("gated", gated),
            ("lse", lse.T),
            ("norm_mid", raw),
        ):
            torch.testing.assert_close(rec[field][0], expected[token], atol=0, rtol=0)
    assert tape.layers[0][0].count_nonzero() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_norm_record(0, 1, x, residual, weight, 1e-6, qkv, q, k, attended)
        fused_silu_record(0, gate_up)
    tape.token_slots[:3] = torch.tensor([7, -1, 3], device="cuda")
    graph.replay()
    rec = restore_values(tape.layer("r", 0, 7, 8))
    torch.testing.assert_close(rec["qkv"][0], qkv[0], atol=0, rtol=0)
    torch.testing.assert_close(rec["gated"][0], gated[0], atol=0, rtol=0)
    assert tape.layers[0].shape[-1] == (304 if compact else 416)
