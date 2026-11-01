import math
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from .hero_objective import advantages, shaped_return, token_objective
from .hero_v2 import HeroConfig, identity, learning_rate, reusable_episode, validate_resume_identity


def config(gpus=8):
    return HeroConfig(model='/models/768f209d9ea81521153ed38c47d515654e938aea',
        source='/native',output='/run',node_dir='/node/control',source_sha256='a'*64,
        deadline=10000,gpus=gpus)


def test_group_normalization_uses_sample_std_and_zero_groups():
    values,std=advantages(list(range(8)))
    assert std == pytest.approx(math.sqrt(6))
    assert sum(values) == pytest.approx(0)
    assert values[-1] == pytest.approx(3.5/(math.sqrt(6)+1e-4))
    assert advantages([.25]*8)==([0.]*8,0.)


def test_native_format_and_invalid_penalties_are_disjoint():
    row=dict(training_return=.5,format_errors=2,invalid_actions=3)
    assert shaped_return(row)==pytest.approx(.4)


def test_kl_zero_at_identical_policy_and_reference():
    logits=torch.tensor([[1.,2.,-1.],[2.,-1.,0.]],requires_grad=True)
    labels=torch.tensor([1,0])
    reference=(logits.detach()/.6).log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
    loss,stats=token_objective(logits,labels,reference,reference,0.,entropy_coefficient=0.)
    loss.backward()
    assert stats[1].item()==pytest.approx(0,abs=1e-6)
    assert logits.grad.norm().item()<1e-6


def test_entropy_gradient_survives_zero_task_advantage():
    logits=torch.tensor([[3.,1.,-2.]],requires_grad=True)
    labels=torch.tensor([0]);reference=(logits.detach()/.6).log_softmax(-1)[:,0]
    loss,stats=token_objective(logits,labels,reference,reference,0.,kl_coefficient=0.,entropy_coefficient=.01)
    loss.backward()
    assert stats[2].item()>0 and logits.grad[0,0]>0 and logits.grad[0,1]<0


def test_reference_kl_is_not_weighted_by_reward_advantage():
    logits=torch.tensor([[1.,0.,-1.]],requires_grad=True)
    labels=torch.tensor([0]);old=(logits.detach()/.6).log_softmax(-1)[:,0]
    loss,stats=token_objective(logits,labels,old-.3,old,0.,kl_coefficient=.1,entropy_coefficient=0.)
    loss.backward()
    assert stats[1].item()>0 and logits.grad.norm().item()>0


def test_positive_advantage_increases_chosen_probability():
    logits=torch.tensor([[.1,.2,.3]],requires_grad=True);labels=torch.tensor([1])
    old=(logits.detach()/.6).log_softmax(-1)[:,1]
    loss,_=token_objective(logits,labels,old,old,1.,kl_coefficient=0.,entropy_coefficient=0.)
    loss.backward()
    assert logits.grad[0,1]<0


def test_ppo_clips_large_positive_ratio():
    logits=torch.tensor([[.1,.2,.3]],requires_grad=True);labels=torch.tensor([1])
    current=(logits.detach()/.6).log_softmax(-1)[:,1]
    loss,stats=token_objective(logits,labels,current,current-1.,1.,kl_coefficient=0.,entropy_coefficient=0.)
    loss.backward()
    assert logits.grad.norm().item()==pytest.approx(0,abs=1e-6)
    assert stats[4].item()==1


def test_unselected_positions_get_no_direct_gradient():
    logits=torch.randn(6,5,requires_grad=True);mask=torch.tensor([False,True,False,True,False,True])
    selected=logits[mask];labels=torch.tensor([1,2,3]);old=(selected.detach()/.6).log_softmax(-1).gather(-1,labels[:,None]).squeeze(-1)
    loss,_=token_objective(selected,labels,old,old,1.)
    loss.backward()
    assert logits.grad[~mask].abs().sum().item()==0
    assert logits.grad[mask].abs().sum().item()>0


def test_four_and_eight_gpu_layouts_keep_ml_identity_and_full_batch_capacity():
    eight,four=config(8),config(4)
    assert eight.batch_size==four.batch_size==128
    assert eight.live_games==32 and four.live_games==64
    assert eight.live_games*eight.actors==four.live_games*four.actors==128
    assert identity(eight)==identity(four)


def test_continuations_preserve_parent_schedule_and_terminal_lr():
    parent = config()
    continued = replace(parent, updates=192, schedule_updates=128)
    for update in (1, 8, 32, 64, 127, 128):
        assert learning_rate(continued, update) == learning_rate(parent, update)
    assert learning_rate(continued, 129) == pytest.approx(4e-7)
    assert learning_rate(continued, 192) == learning_rate(continued, 128)
    assert identity(continued)['lr_schedule_updates'] == 128
    small = replace(continued, window=256, gpus=4)
    assert learning_rate(small, 192) == learning_rate(continued, 192)
    assert identity(small) | {'window': 1024} == identity(replace(continued, gpus=4))


def test_completed_episode_reuse_requires_same_policy_controller_and_unchanged_tokens(tmp_path):
    cfg=config()
    assert reusable_episode(tmp_path/'missing.json',{},cfg) is None
    tokens=tmp_path/'tokens.json.gz';tokens.write_bytes(b'preserved rollout bytes')
    spec=dict(episode_id='u0001-g0-r0',policy_version=0,board_seed=1234)
    row=dict(spec,thinking_token_limit=128,action_token_limit=32,token_record=str(tokens),
        token_record_sha256=hashlib.sha256(tokens.read_bytes()).hexdigest())
    result=tmp_path/'result.json';result.write_text(json.dumps(row))
    assert reusable_episode(result,spec,cfg)==row
    with pytest.raises(ValueError,match='seed/policy'):
        reusable_episode(result,dict(spec,policy_version=1),cfg)
    row['action_token_limit']=64;result.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='controller'):
        reusable_episode(result,spec,cfg)
    row['action_token_limit']=32;result.write_text(json.dumps(row));tokens.write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='tokens changed'):
        reusable_episode(result,spec,cfg)


def test_reviewed_source_migration_preserves_every_other_identity_field():
    original=config();saved=identity(original)
    current=replace(original,source_sha256='b'*64,defer_native_logs=True)
    with pytest.raises(ValueError,match='reviewed source'):
        validate_resume_identity(saved,current)
    current=replace(current,accepted_resume_sources=('a'*64,))
    validate_resume_identity(saved,current)
    validate_resume_identity(identity(current),current)
    for key in saved.keys()-{'source_sha256'}:
        with pytest.raises(ValueError,match='cannot change'):
            validate_resume_identity(saved | {key:'modified'},current)
