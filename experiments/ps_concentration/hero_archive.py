"""Verified lossless checkpoint archives, with optional XOR deltas for BF16 weights."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


BLOCK = 8 * 1024**2
FORMAT = 'ps-hero-zstd-xor-v1'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(BLOCK):
            value.update(block)
    return value.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return s.st_ino, s.st_size, s.st_mtime_ns


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    os.fsync(directory)
    os.close(directory)


def metadata_path(archive):
    return Path(str(archive) + '.json')


def xor(block, reference):
    other = reference.read(len(block))
    if len(other) != len(block):
        raise ValueError('Delta reference has a different size')
    return np.bitwise_xor(np.frombuffer(block, dtype=np.uint8),
                          np.frombuffer(other, dtype=np.uint8)).tobytes()


def decoded(archive, reference=None):
    with ExitStack() as stack:
        ref = stack.enter_context(Path(reference).open('rb')) if reference else None
        process = stack.enter_context(subprocess.Popen(
            ['zstd', '-q', '-d', '-c', str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        with process.stdout:
            while block := process.stdout.read(BLOCK):
                yield xor(block, ref) if ref else block
        if process.wait() != 0:
            raise ValueError('Archive decoding failed: ' + process.stderr.read().decode(errors='replace'))
        if ref and ref.read(1):
            raise ValueError('Delta reference has trailing bytes')


def verify(archive, *, reference=None, metadata=None):
    archive = Path(archive)
    meta = metadata or json.loads(metadata_path(archive).read_text())
    if meta['format'] != FORMAT:
        raise ValueError('Unknown archive format')
    expected_ref = meta['reference']
    if bool(reference) != bool(expected_ref):
        raise ValueError('Supply the exact delta reference when required')
    if reference and digest(reference) != expected_ref['sha256']:
        raise ValueError('Delta reference hash changed')
    if digest(archive) != meta['archive_sha256']:
        raise ValueError('Compressed archive hash changed')
    value, size = hashlib.sha256(), 0
    for block in decoded(archive, reference):
        value.update(block)
        size += len(block)
    if (size, value.hexdigest()) != (meta['source_bytes'], meta['source_sha256']):
        raise ValueError('Archive round trip differs from its source')
    return meta


def archive(source, destination, *, reference=None, expected_sha256=None):
    source, destination = Path(source), Path(destination)
    reference = Path(reference) if reference else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or metadata_path(destination).exists():
        meta = verify(destination, reference=reference)
        if digest(source) != meta['source_sha256']:
            raise ValueError('Existing archive belongs to a different source')
        return meta
    before = stamp(source)
    ref_before = stamp(reference) if reference else None
    if reference and before[1] != ref_before[1]:
        raise ValueError('Delta inputs must have equal lengths')
    temporary = destination.with_name(destination.name + f'.tmp-{os.getpid()}')
    value, ref_hash = hashlib.sha256(), hashlib.sha256()
    with ExitStack() as stack:
        src = stack.enter_context(source.open('rb'))
        ref = stack.enter_context(reference.open('rb')) if reference else None
        process = stack.enter_context(subprocess.Popen(
            ['zstd', '-q', '-T2', '-3', '-o', str(temporary)], stdin=subprocess.PIPE, stderr=subprocess.PIPE))
        with process.stdin:
            while block := src.read(BLOCK):
                value.update(block)
                if ref:
                    other = ref.read(len(block))
                    if len(other) != len(block):
                        raise ValueError('Delta reference changed size')
                    ref_hash.update(other)
                    block = np.bitwise_xor(np.frombuffer(block, dtype=np.uint8),
                                           np.frombuffer(other, dtype=np.uint8)).tobytes()
                process.stdin.write(block)
        if process.wait() != 0:
            raise RuntimeError('Archive encoding failed: ' + process.stderr.read().decode(errors='replace'))
    if stamp(source) != before or (reference and stamp(reference) != ref_before):
        raise ValueError('Archive input changed while encoding')
    if expected_sha256 and value.hexdigest() != expected_sha256:
        raise ValueError('Source differs from its previously verified hash')
    meta = dict(format=FORMAT, source=str(source), source_bytes=before[1], source_sha256=value.hexdigest(),
        reference=dict(path=str(reference), sha256=ref_hash.hexdigest(), bytes=ref_before[1]) if reference else None,
        archive_bytes=temporary.stat().st_size, archive_sha256=digest(temporary),
        zstd_version=subprocess.check_output(['zstd', '--version'], text=True).strip(),
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    verify(temporary, reference=reference, metadata=meta)
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    write_json(metadata_path(destination), meta)
    return meta


def restore(archive_path, destination, *, reference=None):
    archive_path, destination = Path(archive_path), Path(destination)
    meta = verify(archive_path, reference=reference)
    if destination.exists():
        if digest(destination) != meta['source_sha256']:
            raise ValueError('Restore destination already contains different data')
        return meta
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f'.tmp-{os.getpid()}')
    with temporary.open('xb') as handle:
        for block in decoded(archive_path, reference):
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    if digest(temporary) != meta['source_sha256']:
        raise ValueError('Restored file hash differs')
    temporary.replace(destination)
    return meta


def retire_duplicate(source, archive_path, *, reference=None):
    source = Path(source)
    meta = verify(archive_path, reference=reference)
    before = stamp(source)
    if digest(source) != meta['source_sha256'] or stamp(source) != before:
        raise ValueError('Only an unchanged, verified duplicate may be retired')
    write_json(Path(str(source) + '.restore.json'), dict(archive=str(archive_path),
        archive_metadata=str(metadata_path(archive_path)), reference=str(reference) if reference else None,
        source_sha256=meta['source_sha256'], source_bytes=meta['source_bytes']))
    source.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['archive', 'restore', 'verify'])
    parser.add_argument('--source')
    parser.add_argument('--archive', required=True)
    parser.add_argument('--destination')
    parser.add_argument('--reference')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--retire-source', action='store_true')
    args = parser.parse_args()
    if args.operation == 'archive':
        if not args.source:
            parser.error('archive requires --source')
        result = archive(args.source, args.archive, reference=args.reference, expected_sha256=args.expected_sha256)
        if args.retire_source:
            retire_duplicate(args.source, args.archive, reference=args.reference)
    elif args.operation == 'restore':
        if not args.destination:
            parser.error('restore requires --destination')
        result = restore(args.archive, args.destination, reference=args.reference)
    else:
        result = verify(args.archive, reference=args.reference)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
