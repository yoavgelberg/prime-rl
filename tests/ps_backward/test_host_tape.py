from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.ps_host_tape import PinnedActivationTape, RowCopies
from prime_rl.trainer.ps_backward.retirement import RetirementSchedule


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["dma", "sm"])
def test_strided_dma_copies_preserve_columns_and_padding(mode):
    copy = RowCopies(mode)
    device = torch.arange(11 * 19, dtype=torch.float32, device="cuda").view(11, 19)
    host = torch.full((11, 23), -1.0, pin_memory=True)
    copy(host[:, 5:12], device[:, 3:10])
    destination = torch.zeros((11, 13), device="cuda")
    copy(destination[:, 2:9], host[:, 5:12])
    torch.cuda.synchronize()
    torch.testing.assert_close(destination[:, 2:9], device[:, 3:10])
    assert (host[:, :5] == -1).all() and (host[:, 12:] == -1).all()
    assert (destination[:, :2] == 0).all() and (destination[:, 9:] == 0).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reuse_gpu,independent_reads", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("mode", ["dma", "sm"])
def test_pinned_retirement_survives_many_ring_wraps_and_reuses_all_pools(reuse_gpu, independent_reads, mode):
    depth, chunk, prefix, window, capacity = 3, 8, 5, 13, 24
    source = SimpleNamespace(
        depth=depth,
        hidden=4,
        qwidth=4,
        kwidth=2,
        slots=1,
        capacity=capacity,
        activation_dtype=torch.float32,
        policy_version=7,
        requests={"r": 0},
        fields=(("query", 4), ("key", 2), ("qkv", 8)),
    )
    source.layers = [torch.empty((capacity, 14), device="cuda") for _ in range(depth)]
    source.norm_inputs = [torch.empty((capacity, 4), device="cuda") for _ in range(2 * depth + 1)]
    source.lse = [torch.empty((capacity, 2), device="cuda") for _ in range(depth)]
    source.final = torch.empty((capacity, 8), device="cuda")
    source.logits = torch.empty((capacity, 17), device="cuda")
    inputs = torch.empty((capacity, 1), device="cuda", dtype=torch.long)
    targets = torch.empty_like(inputs)
    tape = PinnedActivationTape(
        source, chunk, window, lambda: None, reuse_gpu=reuse_gpu, copy_mode=mode, independent_reads=independent_reads
    )
    schedule = RetirementSchedule(depth, window, prefix, chunk, 7)
    worker = torch.cuda.Stream()
    runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["r"]), seq_lens=SimpleNamespace(np=[0]))
    checks = []

    def consume(action, block, layer):
        if action == "query":
            rec = tape.load_layer("r", layer, block.start, block.end, torch.device("cuda"))
            expected = torch.arange(block.start, block.end, device="cuda", dtype=torch.float32)
            checks.append((rec["query"][:, 0].clone(), expected + layer * 10000))
            if block.end - block.start > 1:
                part = tape.load_layer(
                    "r",
                    layer,
                    block.start + 1,
                    block.end,
                    torch.device("cuda"),
                    ("key",),
                    copy_mode="sm" if mode == "dma" else "dma",
                )
                checks.append((part["key"][:, 0].clone(), expected[1:] + layer * 10000))
        elif action == "release_layer":
            tape.release_layer("r", layer, block.start, block.end)

    for end in [prefix, *range(prefix + chunk, 1029 + 1, chunk)]:
        start = schedule.end
        runner.seq_lens.np[0] = end
        tape.before_write(runner)
        rows = torch.arange(start, end, device="cuda")
        indices = rows % capacity
        for layer, buffer in enumerate(source.layers):
            buffer[indices] = (rows + layer * 10000)[:, None].float()
        for buffer in [*source.norm_inputs, *source.lse, source.final, source.logits]:
            buffer[indices] = rows[:, None].float()
        inputs[indices], targets[indices] = rows[:, None], rows[:, None] + 1
        ready = torch.cuda.Event()
        ready.record()
        published = tape.publish("r", start, end, ready, inputs, targets)
        with torch.cuda.stream(worker):
            worker.wait_event(published)
            head = tape.head("r", start, end, 0)
            ids, labels = tape.tokens("r", start)
            checks.extend(((ids, rows), (labels, rows + 1)))
            checks.append((head["raw"].to("cuda", non_blocking=True)[:, 0], rows.float()))
            tape.release_head("r", start, end)
            schedule.append(end, policy_version=7)
            schedule.advance(consume, policy_version=7)
    with torch.cuda.stream(worker):
        schedule.advance(consume, policy_version=7, finished=True)
    torch.cuda.synchronize()
    for actual, expected in checks:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    report = tape.report()
    assert report["live_layer_blocks"] == report["live_head_blocks"] == 0
    assert report["gpu_ring_reuse_waits"] > 10
    assert report["dma_bytes"]["h2d"] > 0
    assert (report["gpu_ring_cache_hits"] > 0) == reuse_gpu
    assert report["independent_ring_copies"] == independent_reads
    assert report["bounded_activation_storage"]
    for pool in [*tape.layer_pools, tape.head_pool]:
        assert len(pool.free) == pool.slots
    tape.reset(lambda: None)
