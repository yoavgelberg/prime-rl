import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from .hero_archive import digest
from .hero_policy_archive import hydrate, preserve, verify_policy


pytestmark = pytest.mark.skipif(shutil.which('zstd') is None, reason='System zstd is required')


def catalog(directory, values, version):
    directory.mkdir()
    (directory / 'model.safetensors').write_bytes(values.tobytes())
    (directory / 'config.json').write_text(json.dumps({'fixture_version': version}))
    return dict(status='VERIFIED', policy=str(directory), policy_version=version,
        files=[dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(directory.iterdir())])


def test_policy_archive_survives_original_export_removal(tmp_path):
    values = np.arange(20000, dtype=np.uint16)
    base = catalog(tmp_path / 'base-source', values, 128)
    changed = values.copy()
    changed[::13] ^= 3
    child = catalog(tmp_path / 'child-source', changed, 144)
    base_archive, child_archive = tmp_path / 'base-archive', tmp_path / 'child-archive'
    preserve(base, base_archive)
    preserve(child, child_archive, reference=base_archive)
    shutil.rmtree(base['policy'])
    shutil.rmtree(child['policy'])
    target = hydrate(child_archive, tmp_path / 'cache')
    assert (target / 'model.safetensors').read_bytes() == changed.tobytes()
    assert json.loads((target / 'config.json').read_text()) == {'fixture_version': 144}
    assert hydrate(child_archive, target) == target
    assert preserve(child, child_archive, reference=base_archive)['catalog'] == child


def test_policy_delta_catalog_cannot_claim_different_weights(tmp_path):
    values = np.arange(1000, dtype=np.uint16)
    base = catalog(tmp_path / 'base', values, 128)
    child = catalog(tmp_path / 'child', values ^ 1, 144)
    preserve(base, tmp_path / 'base-archive')
    preserve(child, tmp_path / 'child-archive', reference=tmp_path / 'base-archive')
    manifest = tmp_path / 'child-archive/archive-manifest.json'
    data = json.loads(manifest.read_text())
    next(r for r in data['catalog']['files'] if r['name'] == 'model.safetensors')['sha256'] = '0' * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='payload and export catalog'):
        verify_policy(manifest.parent)
