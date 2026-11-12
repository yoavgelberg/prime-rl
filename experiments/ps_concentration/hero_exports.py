"""Verify hero inference exports against their optimizer snapshot on CPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stamp(path):
    state = Path(path).stat()
    return dict(inode=state.st_ino, bytes=state.st_size, mtime_ns=state.st_mtime_ns)


def verify_tokenizer(reference, policy):
    from transformers import AutoTokenizer

    source = AutoTokenizer.from_pretrained(reference, local_files_only=True)
    exported = AutoTokenizer.from_pretrained(policy, local_files_only=True)
    if source.get_vocab() != exported.get_vocab() or source.get_chat_template() != exported.get_chat_template():
        raise ValueError('Exported tokenizer vocabulary or native chat template changed')
    if source.all_special_ids != exported.all_special_ids or any(
        getattr(source, key) != getattr(exported, key) for key in ('bos_token_id', 'eos_token_id', 'pad_token_id')):
        raise ValueError('Exported tokenizer special tokens changed')
    return dict(vocabulary_size=len(source), native_chat_template_equal=True, special_tokens_equal=True)


def verify(checkpoint, policy, output, *, expected_parameters=4022468096):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(2)
    checkpoint, policy, output = map(Path, (checkpoint, policy, output))
    if output.resolve().is_relative_to(policy.resolve()):
        raise ValueError('Store verification receipts outside the export directory')
    original = stamp(checkpoint)
    sidecar = read(checkpoint.with_suffix(checkpoint.suffix + '.json'))
    metadata = read(policy / 'experiment.json')
    if sidecar['bytes'] != original['bytes']:
        raise ValueError('Snapshot file does not match its committed sidecar')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if saved['format'] != 'ps-hero-v2-adamw-v1' or saved['identity'].get('variant') != 'hero-v2':
        raise ValueError('Use a full hero model/optimizer snapshot')
    version, identity = saved['policy_version'], saved['identity']
    if version != sidecar['policy_version'] or version != metadata['policy_version'] or version <= 0:
        raise ValueError('Snapshot and exported policy versions differ')
    if identity != sidecar['identity'] or any(metadata.get(key) != value for key, value in identity.items()):
        raise ValueError('Snapshot and exported training identities differ')
    reference = Path(identity['reference_checkpoint'])
    if reference.name != identity['model_revision']:
        raise ValueError('The reference checkpoint must retain its immutable revision')
    base_config, exported_config = read(reference / 'config.json'), read(policy / 'config.json')
    config_fields = ('model_type', 'hidden_size', 'head_dim', 'num_hidden_layers', 'num_attention_heads',
        'num_key_value_heads', 'intermediate_size', 'vocab_size', 'hidden_act', 'attention_bias',
        'tie_word_embeddings', 'rms_norm_eps', 'rope_theta', 'rope_scaling', 'max_position_embeddings',
        'use_sliding_window', 'sliding_window', 'bos_token_id', 'eos_token_id')
    if any(base_config.get(key) != exported_config.get(key) for key in config_fields):
        raise ValueError('Exported architecture, RoPE, context or stop-token configuration changed')
    aliases, unique = {}, {}
    for name, value in saved['model'].items():
        if value.dtype != torch.float32 or not value.is_contiguous():
            raise ValueError('Expected contiguous FP32 master weights in the snapshot')
        key = value.untyped_storage().data_ptr(), tuple(value.shape), tuple(value.stride())
        if key in aliases:
            continue
        aliases[key] = name
        unique[name] = value
    parameters = sum(value.numel() for value in unique.values())
    if parameters != expected_parameters:
        raise ValueError('Snapshot unique parameter count differs from the expected model')
    tensor_rows = []
    weights_stamp = stamp(policy / 'model.safetensors')
    with safe_open(policy / 'model.safetensors', framework='pt', device='cpu') as tensors:
        if set(tensors.keys()) != set(unique):
            raise ValueError('Exported tensor names or tied-weight aliases changed')
        for name, value in unique.items():
            actual = tensors.get_tensor(name)
            if actual.dtype != torch.bfloat16 or actual.shape != value.shape:
                raise ValueError(f'Exported tensor shape or dtype changed: {name}')
            flattened, expected = actual.view(-1), value.view(-1)
            for start in range(0, value.numel(), 8 * 1024**2):
                if not torch.equal(flattened[start:start + 8 * 1024**2],
                                   expected[start:start + 8 * 1024**2].bfloat16()):
                    raise ValueError(f'Export differs from the effective BF16 training weights: {name}')
            tensor_rows.append(dict(name=name, shape=list(value.shape), elements=value.numel()))
    tokenizer = verify_tokenizer(reference, policy)
    generation_source = reference / 'generation_config.json'
    generation_target = policy / 'generation_config.json'
    supplied_generation_config = False
    if generation_source.exists():
        if not generation_target.exists():
            temporary = generation_target.with_suffix('.json.tmp')
            shutil.copyfile(generation_source, temporary)
            temporary.replace(generation_target)
            supplied_generation_config = True
        if read(generation_target) != read(generation_source):
            raise ValueError('Exported generation configuration differs from the reference')
    if stamp(checkpoint) != original or read(checkpoint.with_suffix(checkpoint.suffix + '.json')) != sidecar:
        raise ValueError('Snapshot was replaced during verification; retry against its matching export')
    files = []
    for path in sorted(policy.iterdir()):
        if path.is_file() and '.tmp' not in path.name:
            before = stamp(path)
            digest = file_hash(path)
            if stamp(path) != before:
                raise ValueError('Exported file changed while hashing')
            files.append(dict(name=path.name, bytes=before['bytes'], sha256=digest))
    if (stamp(checkpoint) != original or read(checkpoint.with_suffix(checkpoint.suffix + '.json')) != sidecar
            or stamp(policy / 'model.safetensors') != weights_stamp or read(policy / 'experiment.json') != metadata):
        raise ValueError('Snapshot or exported policy changed during verification')
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), status='VERIFIED', policy_version=version,
        policy=str(policy), snapshot=str(checkpoint), snapshot_stamp=original, snapshot_sidecar=sidecar,
        identity=identity, unique_parameters=parameters, tensors=tensor_rows, tokenizer=tokenizer,
        generation_config_supplied_from_reference=supplied_generation_config, files=files, node=socket.gethostname(),
        scope='Every exported BF16 parameter equals conversion of the saved FP32 master weights. '
            'CPU-only verification; no inference, backward, optimizer restore or checkpoint selection. '
            'The policy remains node-local until explicitly preserved elsewhere.')
    write(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.policy, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'policy_version', 'unique_parameters', 'policy')}))


if __name__ == '__main__':
    main()
