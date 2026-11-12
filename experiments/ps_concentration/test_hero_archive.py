import hashlib
import shutil

import numpy as np
import pytest

from .hero_archive import archive, digest, restore, retire_duplicate, verify


pytestmark = pytest.mark.skipif(shutil.which('zstd') is None, reason='System zstd is required')


def test_verified_archive_can_retire_and_restore_exact_optimizer_bytes(tmp_path):
    import torch

    source = tmp_path / 'checkpoint.pt'
    state = dict(model=torch.arange(128, dtype=torch.float32),
                 optimizer=dict(step=128, exp_avg=torch.randn(128), exp_avg_sq=torch.rand(128)),
                 rng=torch.get_rng_state())
    torch.save(state, source)
    expected = digest(source)
    target = tmp_path / 'checkpoint.pt.zst'
    meta = archive(source, target, expected_sha256=expected)
    retire_duplicate(source, target)
    assert not source.exists() and (tmp_path / 'checkpoint.pt.restore.json').exists()
    restore(target, source)
    assert digest(source) == expected == meta['source_sha256']
    actual = torch.load(source, weights_only=True)
    assert torch.equal(actual['optimizer']['exp_avg'], state['optimizer']['exp_avg'])
    assert torch.equal(actual['rng'], state['rng'])


def test_xor_archive_requires_exact_reference_and_restores_changed_weights(tmp_path):
    reference, source = tmp_path / 'base.bin', tmp_path / 'weights.bin'
    base = np.arange(100_000, dtype=np.uint16)
    changed = base.copy()
    changed[::17] ^= 3
    reference.write_bytes(base.tobytes())
    source.write_bytes(changed.tobytes())
    target = tmp_path / 'weights.xor.zst'
    meta = archive(source, target, reference=reference)
    assert meta['archive_bytes'] < meta['source_bytes'] / 10
    with pytest.raises(ValueError, match='reference'):
        verify(target)
    wrong = tmp_path / 'wrong.bin'
    wrong.write_bytes(bytes(reference.stat().st_size))
    with pytest.raises(ValueError, match='reference hash'):
        restore(target, tmp_path / 'bad.bin', reference=wrong)
    restored = tmp_path / 'restored.bin'
    restore(target, restored, reference=reference)
    assert restored.read_bytes() == source.read_bytes()


def test_archive_corruption_cannot_retire_source(tmp_path):
    source, target = tmp_path / 'data', tmp_path / 'data.zst'
    source.write_bytes(b'complete experiment state' * 100)
    archive(source, target)
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 1
    target.write_bytes(payload)
    with pytest.raises(ValueError, match='archive hash'):
        retire_duplicate(source, target)
    assert source.is_file()


def test_changed_source_is_preserved_and_not_replaced(tmp_path):
    source, target = tmp_path / 'data', tmp_path / 'data.zst'
    source.write_bytes(b'original')
    archive(source, target)
    source.write_bytes(b'new scientific result')
    with pytest.raises(ValueError, match='verified duplicate'):
        retire_duplicate(source, target)
    with pytest.raises(ValueError, match='different source'):
        archive(source, target)
    assert source.read_bytes() == b'new scientific result'


def test_expected_source_hash_failure_publishes_no_archive(tmp_path):
    source, target = tmp_path / 'data', tmp_path / 'data.zst'
    source.write_bytes(b'not the pinned artifact')
    with pytest.raises(ValueError, match='previously verified hash'):
        archive(source, target, expected_sha256=hashlib.sha256(b'pinned').hexdigest())
    assert source.is_file() and not target.exists()


def test_existing_restore_target_is_never_overwritten(tmp_path):
    source, target, restored = tmp_path / 'source', tmp_path / 'data.zst', tmp_path / 'existing'
    source.write_bytes(b'preserved')
    archive(source, target)
    restored.write_bytes(b'another result')
    with pytest.raises(ValueError, match='different data'):
        restore(target, restored)
    assert restored.read_bytes() == b'another result'
