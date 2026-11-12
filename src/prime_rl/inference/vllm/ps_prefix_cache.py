"""Protect instruction-prefix KV pages independently of prefix-cache reuse."""

from functools import wraps


def protect_prefix(coordinator, request):
    for manager in coordinator.single_type_managers:
        if not hasattr(manager, 'pinned_blocks_by_req'):
            continue
        blocks = (request.num_prompt_tokens + manager.block_size - 1) // manager.block_size
        blocks = max(manager.pinned_blocks_env, blocks)
        previous = manager.pinned_blocks_by_req.get(request.request_id)
        if previous is not None and previous != blocks:
            raise ValueError('A live PS request cannot change its pinned prefix')
        manager.pinned_blocks_by_req[request.request_id] = blocks


def install(manager_class=None):
    if manager_class is None:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        manager_class = KVCacheManager
    if getattr(manager_class, '_ps_prefix_retention_installed', False):
        return
    allocate, free = manager_class.allocate_slots, manager_class.free

    @wraps(allocate)
    def allocate_slots(self, request, *args, **kwargs):
        protect_prefix(self.coordinator, request)
        return allocate(self, request, *args, **kwargs)

    @wraps(free)
    def free_request(self, request):
        result = free(self, request)
        for manager in self.coordinator.single_type_managers:
            if hasattr(manager, 'pinned_blocks_by_req'):
                manager.pinned_blocks_by_req.pop(request.request_id, None)
        return result

    manager_class.allocate_slots = allocate_slots
    manager_class.free = free_request
    manager_class._ps_prefix_retention_installed = True
