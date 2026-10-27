import pytest
import torch

from prime_rl.trainer.ps_backward.fa4 import AlignedChunkPlans, aligned_chunk_plan


@pytest.mark.parametrize("prefix,window,chunk", [(7, 3, 1), (69, 2048, 512), (153, 2048, 2048), (257, 511, 256)])
def test_shifted_layout_matches_independently_rebuilt_sparse_tiles(prefix, window, chunk):
    cache = AlignedChunkPlans()
    for block in range(20):
        start = prefix + block * chunk
        for count in (chunk, max(1, chunk - 1)):
            end, history = start + count, max(prefix, start - window + 1)
            actual = cache.get(prefix, start, end, history, window, "cpu", (256, 128))
            expected = aligned_chunk_plan(prefix, start, end, history, window, "cpu", (256, 128))
            for field in ("q_indices", "k_indices", "source_indices"):
                if getattr(expected, field) is None:
                    assert getattr(actual, field) is None
                else:
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
            for left, right in zip(
                (*actual.forward, *actual.backward), (*expected.forward, *expected.backward), strict=True
            ):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    if chunk >= 256:
        assert len(cache.templates) < 25
