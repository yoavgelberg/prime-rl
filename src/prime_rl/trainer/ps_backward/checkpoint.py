"""Atomic FP32 optimizer snapshots and native-BF16 inference checkpoints."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def model_state(model, *, dtype=None):
    shared, result = {}, {}
    for name, value in model.state_dict().items():
        key = value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype, value.device
        if key not in shared:
            target_dtype = dtype if dtype is not None and value.is_floating_point() else value.dtype
            shared[key] = value.detach().to(device='cpu', dtype=target_dtype, copy=True)
        result[name] = shared[key]
    return result


def save_snapshot(path, model, optimizer_state, *, policy_version, outcome_config, rngs, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{policy_version}')
    if temporary.exists():
        temporary.rename(temporary.with_name(temporary.name + f'.interrupted-{time.time_ns()}'))
    state = dict(format='prime-rl-recorded-grpo-v1', model=model_state(model),
        optimizer=cpu_tree(optimizer_state), policy_version=policy_version,
        outcome_config=asdict(outcome_config), rngs=cpu_tree(rngs), metadata=metadata)
    torch.save(state, temporary)
    os.replace(temporary, path)
    receipt = dict(format=state['format'], policy_version=policy_version, bytes=path.stat().st_size,
                   outcome_config=state['outcome_config'], metadata=metadata)
    sidecar = path.with_suffix(path.suffix + '.json')
    temporary = sidecar.with_name(sidecar.name + '.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    os.replace(temporary, sidecar)
    return receipt


def restore_snapshot(path, model, optimizer, *, outcome_config, expected_metadata):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'prime-rl-recorded-grpo-v1' or state['outcome_config'] != asdict(outcome_config):
        raise ValueError('Snapshot algorithm or scientific optimizer configuration changed')
    for name, expected in expected_metadata.items():
        if state['metadata'].get(name) != expected:
            raise ValueError(f'Snapshot identity changed: {name}')
    saved_groups = state['optimizer']['param_groups']
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError('Optimizer parameter groups changed on resume')
    for saved, current in zip(saved_groups, optimizer.param_groups, strict=True):
        for key in ('betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'fused'):
            if saved.get(key) != current.get(key):
                raise ValueError(f'Optimizer setting changed on resume: {key}')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    return state


def save_inference(path, model, tokenizer, *, policy_version, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    dtype_attribute = "dtype" if hasattr(model.config, "dtype") else "torch_dtype"
    previous_dtype = getattr(model.config, dtype_attribute)
    try:
        model.save_pretrained(path, state_dict=model_state(model, dtype=torch.bfloat16),
                              safe_serialization=True, max_shard_size='2GB')
    finally:
        setattr(model.config, dtype_attribute, previous_dtype)
    configuration = path / 'config.json'
    saved = json.loads(configuration.read_text())
    saved[dtype_attribute] = 'bfloat16'
    for key in ('dtype', 'torch_dtype'):
        if key in saved:
            saved[key] = 'bfloat16'
    configuration.write_text(json.dumps(saved, indent=2) + '\n')
    tokenizer.save_pretrained(path)
    receipt = dict(metadata, policy_version=policy_version,
        storage='Native vLLM effective BF16 weights, including embeddings and normalization; FP32 masters in optimizer snapshot')
    (path / 'experiment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def save_pending_gradients(path, model, records, *, policy_version, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ValueError('Pending task gradients must cover all trainable parameters')
        gradients[name] = parameter.grad.detach().to(device='cpu', copy=True)
    state = dict(format='prime-rl-pending-task-gradient-v1', policy_version=policy_version,
                 gradients=gradients, records=records, metadata=metadata)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)
    return dict(path=str(path), bytes=path.stat().st_size, policy_version=policy_version,
                episodes=len(records), scope='Task gradient before reference KL; optimizer has not stepped')


def restore_pending_gradients(path, model, *, policy_version, expected_metadata):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format'] != 'prime-rl-pending-task-gradient-v1' or state['policy_version'] != policy_version:
        raise ValueError('Pending task gradient policy version or format changed')
    if state['metadata'] != expected_metadata:
        raise ValueError('Pending task gradient scientific identity changed')
    parameters = dict(model.named_parameters())
    gradients = state['gradients']
    if parameters.keys() != gradients.keys():
        raise ValueError('Pending task gradient parameter names changed')
    if any(gradients[name].shape != parameter.shape or gradients[name].dtype != parameter.dtype
           for name, parameter in parameters.items()):
        raise ValueError('Pending task gradient shape or precision changed')
    for name, parameter in parameters.items():
        parameter.grad = gradients[name].to(device=parameter.device, copy=True)
    return state
