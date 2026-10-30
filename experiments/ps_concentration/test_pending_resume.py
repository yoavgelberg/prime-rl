import asyncio
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from prime_rl.trainer.ps_backward.checkpoint import restore_pending_gradients, save_pending_gradients
from .overlap_config import RunConfig, permutation, training_specs
from . import resume_pending


def fixture(tmp_path):
    config = RunConfig(model='/model/768f209d9ea81521153ed38c47d515654e938aea', source='/native',
        output=str(tmp_path/'ps-seed11'), source_sha256='a'*64, gpus=4, updates=1)
    output = Path(config.output)
    folder = output/'pending-task-gradient/update-0001'
    folder.mkdir(parents=True)
    identity = dict(config.scientific_identity(), evaluation_deferred=True)
    files = []
    for rank in range(config.gpus):
        path = folder/f'rank-{rank}.pt'
        path.write_bytes(b'data')
        files.append(dict(path=str(path), bytes=4, policy_version=0))
    rows = [dict(spec, policy_version=0, score=4, total_pairs=8, max_responses=96,
        training_return=4/8-50/96, response_count=50, done=False, generated_tokens=100,
        raw_transcript_tokens=200, seconds=2., invalid_actions=0, format_errors=0)
        for spec in training_specs(config, 1, permutation(11))]
    materialized = [dict(advantages=[-1.,1.]*8, zero_advantage_groups=0,
        normalized_score_surrogate_loss=.01, clipped_grpo_surrogate_at_origin=0.)]*4
    report = dict(episodes=rows, backward=[dict(peak_allocated_bytes=123,peak_reserved_bytes=456)]*4)
    pending = dict(policy_version=0, identity=identity, files=files, reports=[report], reductions=[],
        materialized=materialized, collection_and_materialize_seconds=.2)
    path = folder.with_suffix('.json')
    path.write_text(json.dumps(pending))
    return config, path, pending


def test_manifest_rejects_wrong_identity_layout_truncation_and_duplicate_episodes(tmp_path):
    config, path, pending = fixture(tmp_path)
    assert resume_pending.validate_manifest(config, path)[0] == pending
    with pytest.raises(ValueError, match='replica count'):
        resume_pending.validate_manifest(replace(config,gpus=8), path)
    changed = dict(pending, identity=dict(pending['identity'],window=512))
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='scientific identity'):
        resume_pending.validate_manifest(config,path)
    path.write_text(json.dumps(pending))
    gradient = Path(pending['files'][0]['path'])
    gradient.write_bytes(b'bad')
    with pytest.raises(ValueError, match='byte count'):
        resume_pending.validate_manifest(config,path)
    gradient.write_bytes(b'data')
    pending['reports'][0]['episodes'][-1]['board_seed'] += 1
    path.write_text(json.dumps(pending))
    with pytest.raises(ValueError, match='fixed plan'):
        resume_pending.validate_manifest(config,path)
    pending['reports'][0]['episodes'][-1]['board_seed'] -= 1
    pending['reports'][0]['episodes'][-1] = pending['reports'][0]['episodes'][0]
    path.write_text(json.dumps(pending))
    with pytest.raises(ValueError, match='64 distinct'):
        resume_pending.validate_manifest(config,path)


def test_pending_gradient_restore_is_exact_and_validates_before_mutation(tmp_path):
    model = torch.nn.Linear(3,2)
    model(torch.ones(2,3)).sum().backward()
    expected = {name:parameter.grad.clone() for name,parameter in model.named_parameters()}
    path=tmp_path/'rank-0.pt'
    save_pending_gradients(path,model,[dict(episode_id='original')],policy_version=2,metadata=dict(seed=11))
    model.zero_grad(set_to_none=True)
    for version,metadata in [(1,dict(seed=11)),(2,dict(seed=22))]:
        with pytest.raises(ValueError):
            restore_pending_gradients(path,model,policy_version=version,expected_metadata=metadata)
        assert all(p.grad is None for p in model.parameters())
    restored=restore_pending_gradients(path,model,policy_version=2,expected_metadata=dict(seed=11))
    assert restored['records']==[dict(episode_id='original')]
    for name,parameter in model.named_parameters():
        assert torch.equal(parameter.grad,expected[name])
    model.zero_grad(set_to_none=True)
    state=torch.load(path,weights_only=True)
    state['gradients']['weight']=torch.ones(5,5)
    torch.save(state,path)
    with pytest.raises(ValueError,match='shape or precision'):
        restore_pending_gradients(path,model,policy_version=2,expected_metadata=dict(seed=11))
    assert all(p.grad is None for p in model.parameters())


def test_pending_resume_commits_original_batch_without_generation(monkeypatch,tmp_path):
    config,path,pending=fixture(tmp_path)
    calls,logged=[],[]

    async def rpc(engine,method,*,kwargs=None,timeout=600.):
        calls.append(method)
        if method=='ps_train_initialize':
            return [dict(world_size=4)]*4
        if method=='ps_train_restore_pending':
            assert kwargs['path']==str(path.with_suffix(''))
            return [dict(episodes=16,policy_version=0)]*4
        if method=='ps_train_reference_kl':
            return [dict(kl_sum=0.,normalized_kl=0.,generated_tokens=1600,
                parity=dict(mean_abs=.001,p99=.01,status='GO'))]*4
        if method=='ps_train_step':
            return [dict(policy_version=1,learning_rate=5e-6/8,gradient_norm=.5,
                weight_update_norm=.001,gradient_clipped=False)]*4
        if method=='ps_train_save':
            return [dict(policy_version=1)]*4
        raise AssertionError('Unexpected generation or update operation: '+method)

    monkeypatch.setattr(resume_pending,'worker_rpc',rpc)
    monkeypatch.setattr(resume_pending,'log_metrics',lambda monitor,prefix,metrics,version:logged.append((metrics,version)))
    receipt=asyncio.run(resume_pending.resume(config,object(),None,path))
    assert receipt['new_trajectories_sampled']==0 and receipt['optimizer_update']==1
    assert calls==['ps_train_initialize','ps_train_restore_pending','ps_train_reference_kl','ps_train_step','ps_train_save']
    assert logged[0][1]==1 and logged[0][0]['episodes']==64
    state=json.loads((Path(config.output)/'state.json').read_text())
    assert state['policy_version']==1 and state['cumulative_generated_tokens']==6400
    saved=json.loads((Path(config.output)/'update-metrics/update-0001.json').read_text())
    assert saved['rollouts']==pending['reports'] and saved['metrics']['resumed_pending_task_gradient']


def test_stale_pending_gradients_cannot_apply_a_second_optimizer_step(monkeypatch,tmp_path):
    config,path,pending=fixture(tmp_path)
    latest=Path(config.output)/'checkpoints/latest.pt'
    latest.parent.mkdir()
    latest.touch()
    calls=[]

    async def rpc(engine,method,*,kwargs=None,timeout=600.):
        calls.append(method)
        if method=='ps_train_initialize':
            return [dict(world_size=4)]*4
        if method=='ps_train_restore':
            return [dict(policy_version=1,metadata={})]*4
        raise AssertionError(method)

    monkeypatch.setattr(resume_pending,'worker_rpc',rpc)
    with pytest.raises(ValueError,match='stale'):
        asyncio.run(resume_pending.resume(config,object(),None,path))
    assert calls==['ps_train_initialize','ps_train_restore']
