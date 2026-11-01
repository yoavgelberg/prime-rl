"""Preserve exact inference exports with a shared full-weight reference."""

import hashlib
import json
import os
from pathlib import Path

from .hero_archive import archive, digest, restore, verify, write_json


def checked_copy(source, destination, expected):
    value = hashlib.sha256()
    with Path(source).open('rb') as src, Path(destination).open('wb') as dst:
        while block := src.read(8 * 1024**2):
            value.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    if value.hexdigest() != expected:
        raise ValueError('Copied export differs from its verified catalog')


def verify_policy(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'archive-manifest.json').read_text())
    if meta.get('status') != 'ARCHIVED' or meta['kind'] not in ('full', 'xor-zstd'):
        raise ValueError('Unrecognized policy archive')
    for row in meta['catalog']['files']:
        if row['name'] == 'model.safetensors' and meta['kind'] == 'xor-zstd':
            payload = verify(directory / 'model.safetensors.xor.zst', reference=Path(meta['reference']) / 'model.safetensors')
            if payload['source_sha256'] != row['sha256'] or payload['source_bytes'] != row['bytes']:
                raise ValueError('Delta payload and export catalog disagree')
        elif digest(directory / row['name']) != row['sha256']:
            raise ValueError('Archived export metadata or full weights changed')
    return meta


def preserve(catalog, destination, *, reference=None):
    destination = Path(destination)
    if catalog.get('status') != 'VERIFIED':
        raise ValueError('Archive only an exactly verified inference export')
    if destination.exists():
        meta = verify_policy(destination)
        if meta['catalog'] != catalog:
            raise ValueError('Existing policy archive belongs to a different export')
        return meta
    if reference:
        reference = Path(reference)
        ref = verify_policy(reference)
        if ref['kind'] != 'full':
            raise ValueError('Use a full reference instead of a chain of dependent deltas')
    temporary = destination.with_name(destination.name + f'.preparing-{os.getpid()}')
    temporary.mkdir(parents=True, exist_ok=False)
    source = Path(catalog['policy'])
    kind = 'xor-zstd' if reference else 'full'
    for row in catalog['files']:
        name = row['name']
        if Path(name).name != name:
            raise ValueError('Export catalog contains a nonlocal filename')
        if name == 'model.safetensors' and reference:
            archive(source / name, temporary / 'model.safetensors.xor.zst',
                reference=reference / name, expected_sha256=row['sha256'])
        else:
            checked_copy(source / name, temporary / name, row['sha256'])
    meta = dict(status='ARCHIVED', kind=kind, catalog=catalog,
        reference=str(reference) if reference else None,
        scope='Exact original model bytes and metadata preserved; no numerical conversion or checkpoint retirement')
    write_json(temporary / 'archive-manifest.json', meta)
    verify_policy(temporary)
    temporary.rename(destination)
    return meta


def hydrate(archive_directory, destination):
    archive_directory, destination = Path(archive_directory), Path(destination)
    meta = verify_policy(archive_directory)
    if destination.exists():
        for row in meta['catalog']['files']:
            if digest(destination / row['name']) != row['sha256']:
                raise ValueError('Existing inference cache has different bytes')
        return destination
    temporary = destination.with_name(destination.name + f'.preparing-{os.getpid()}')
    temporary.mkdir(parents=True, exist_ok=False)
    for row in meta['catalog']['files']:
        name = row['name']
        if name == 'model.safetensors' and meta['kind'] == 'xor-zstd':
            restore(archive_directory / 'model.safetensors.xor.zst', temporary / name,
                reference=Path(meta['reference']) / name)
        else:
            checked_copy(archive_directory / name, temporary / name, row['sha256'])
    temporary.rename(destination)
    return destination
