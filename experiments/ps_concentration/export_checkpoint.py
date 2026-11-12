"""Export a completed optimizer snapshot for evaluation without running validation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from prime_rl.trainer.ps_backward.checkpoint import save_inference
from prime_rl.trainer.ps_backward.outcome import OutcomeConfig


def export_snapshot(checkpoint, output=None, *, reference=None, tokenizer=None):
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if state['format'] != 'prime-rl-recorded-grpo-v1' or state['outcome_config'] != asdict(OutcomeConfig()):
        raise ValueError('Use a recorded-GRPO snapshot with the frozen optimization settings')
    version = int(state['policy_version'])
    if version <= 0:
        raise ValueError('No trained optimizer update is present in this snapshot')
    metadata = state['metadata']
    reference = Path(reference or metadata['reference_checkpoint'])
    if reference.name != metadata['model_revision']:
        raise ValueError('The reference must have the recorded immutable model revision')
    output = Path(output) if output is not None else checkpoint.parent / f'policy-{version:04d}'
    if output.exists():
        raise FileExistsError(f'Preserve the existing export: {output}')
    config = AutoConfig.from_pretrained(reference, local_files_only=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    weights, shared = {}, {}
    for name, value in state['model'].items():
        if not name.startswith('base.'):
            raise ValueError(f'Unexpected model state outside the Qwen adapter: {name}')
        key = value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype
        if key not in shared:
            shared[key] = value.bfloat16() if value.is_floating_point() else value
        weights[name.removeprefix('base.')] = shared[key]
    model.load_state_dict(weights, strict=True, assign=True)
    model.tie_weights()
    if (reference / 'generation_config.json').exists():
        model.generation_config = GenerationConfig.from_pretrained(reference, local_files_only=True)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(reference, local_files_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.tmp-{time.time_ns()}')
    receipt = save_inference(temporary, model, tokenizer, policy_version=version, metadata=dict(metadata,
        inference_export_source=str(checkpoint.resolve()),
        checkpoint_selection='Completed optimizer snapshot; no validation selection performed by export'))
    temporary.rename(output)
    return dict(checkpoint=str(output), policy_version=version, metadata=receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference', type=Path, help='Relocated copy of the recorded immutable starting revision')
    args = parser.parse_args()
    print(json.dumps(export_snapshot(args.checkpoint, args.output, reference=args.reference), indent=2))


if __name__ == '__main__':
    main()
