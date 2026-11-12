from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.ps_prefix_cache import install, protect_prefix


def test_prefix_is_registered_before_uncached_allocation_and_released_after_free():
    sliding = SimpleNamespace(block_size=128, pinned_blocks_env=0, pinned_blocks_by_req={})
    class Manager:
        def __init__(self):
            self.enable_caching = False
            self.coordinator = SimpleNamespace(single_type_managers=(sliding, SimpleNamespace()))

        def allocate_slots(self, request, count):
            assert sliding.pinned_blocks_by_req[request.request_id] == 3
            return count

        def free(self, request):
            assert sliding.pinned_blocks_by_req[request.request_id] == 3

    install(Manager)
    install(Manager)
    manager = Manager()
    request = SimpleNamespace(request_id='game', num_prompt_tokens=342)
    assert manager.allocate_slots(request, 342) == 342
    assert manager.allocate_slots(request, 1) == 1
    manager.free(request)
    assert sliding.pinned_blocks_by_req == {}


def test_live_request_cannot_change_its_prefix_and_requests_are_independent():
    sliding = SimpleNamespace(block_size=128, pinned_blocks_env=0, pinned_blocks_by_req={})
    coordinator = SimpleNamespace(single_type_managers=(sliding,))
    protect_prefix(coordinator, SimpleNamespace(request_id='a', num_prompt_tokens=342))
    protect_prefix(coordinator, SimpleNamespace(request_id='b', num_prompt_tokens=128))
    assert sliding.pinned_blocks_by_req == {'a': 3, 'b': 1}
    with pytest.raises(ValueError, match='cannot change'):
        protect_prefix(coordinator, SimpleNamespace(request_id='a', num_prompt_tokens=512))
