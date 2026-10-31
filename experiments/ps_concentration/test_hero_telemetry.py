import json
import shutil

import pytest

from .hero_telemetry import preserve


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(params=[2, 4])
def run_with_actor_reports(tmp_path, request):
    actors = request.param
    run, node = tmp_path / 'run', tmp_path / 'node'
    save(run / 'state.json', dict(policy_version=1))
    ids = []
    for rank in range(actors):
        rows = [dict(episode_id=f'game-{rank}-{i}', actor_rank=rank, policy_version=0,
            generated_tokens=100, raw_transcript_tokens=200) for i in range(128 // actors)]
        ids.extend(row['episode_id'] for row in rows)
        save(node / f'actor-{rank}/finished-0001.json', dict(episodes=rows,
            seconds=3., devices=[dict(rank=0, kv_pool_allocated_bytes=4096)], reused_episodes=[]))
    save(run / 'update-metrics/update-0001.json', dict(episode_ids=ids,
        metrics=dict(update=1, episodes=128, generated_tokens=12800, raw_transcript_tokens=25600),
        actor_weights=dict(path=str(node / 'weights-0001.pt')), steps=[{} for _ in range(actors)]))
    return run, node, actors


def test_actor_reports_survive_node_cleanup_and_keep_kv_scope(run_with_actor_reports):
    run, node, actors = run_with_actor_reports
    result = preserve(run)
    assert result['episodes'] == 128 and result['actor_reports'] == actors
    shutil.rmtree(node)
    assert preserve(run) == result
    receipt = json.loads((run / 'actor-telemetry/manifest.json').read_text())
    assert len(receipt['updates'][0]['devices']) == actors
    assert 'not per-game retained KV' in receipt['scope']


@pytest.mark.parametrize('change,message', [
    (dict(policy_version=1), 'another collection policy'),
    (dict(episode_id='other-game'), 'exactly once'),
    (dict(generated_tokens=99), 'generated_tokens'),
])
def test_actor_reports_cannot_silently_mix_policy_or_counts(run_with_actor_reports, change, message):
    run, node, _ = run_with_actor_reports
    path = node / 'actor-0/finished-0001.json'
    report = json.loads(path.read_text())
    report['episodes'][0].update(change)
    save(path, report)
    with pytest.raises(ValueError, match=message):
        preserve(run)


def test_preserved_source_report_is_immutable(run_with_actor_reports):
    run, node, _ = run_with_actor_reports
    preserve(run)
    path = node / 'actor-0/finished-0001.json'
    save(path, json.loads(path.read_text()) | dict(seconds=4.))
    with pytest.raises(ValueError, match='differs from its node-local source'):
        preserve(run)
