import torch

from .linear import ProjectionCache


def test_checkpoint_wrappers_share_one_weight_cast():
    weight = torch.nn.Parameter(torch.randn(13, 7))
    cache = ProjectionCache()
    expected = cache.cast(weight)
    for _ in range(100):
        wrapper = weight.detach()
        assert wrapper is not weight
        assert cache.cast(wrapper) is expected
    assert len(cache.casts) == 1


def test_optimizer_mutation_invalidates_cast():
    weight = torch.nn.Parameter(torch.randn(13, 7))
    cache = ProjectionCache()
    old = cache.cast(weight)
    with torch.no_grad():
        weight.add_(1)
    current = cache.cast(weight)
    assert current is not old
    assert torch.equal(current, weight.bfloat16())
    assert len(cache.casts) == 1


def test_different_layouts_and_storages_are_distinct():
    weight = torch.randn(13, 7)
    cache = ProjectionCache()
    assert cache.cast(weight).shape == (13, 7)
    assert cache.cast(weight.T).shape == (7, 13)
    assert cache.cast(weight.clone()).shape == (13, 7)
    assert len(cache.casts) == 3
