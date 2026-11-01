import pytest
import torch

from .hero_objective import token_objective


def test_exact_categorical_kl_value_and_gradient():
    temperature=.6
    logits=torch.tensor([[.4,-.8,1.2],[1.,.2,-.4]],requires_grad=True)
    reference=torch.tensor([[.1,.2,.3],[-.2,.1,.4]]).log_softmax(-1)
    labels=torch.tensor([2,0])
    logp=(logits.detach()/temperature).log_softmax(-1)
    old=logp.gather(-1,labels[:,None]).squeeze(-1)
    loss,stats=token_objective(logits,labels,reference,old,0.,temperature=temperature,
        kl_coefficient=1.,entropy_coefficient=0.)
    loss.backward()
    p=logp.exp();per_token=(p*(logp-reference)).sum(-1)
    expected=p*(logp-reference-per_token[:,None])/temperature
    torch.testing.assert_close(loss.detach(),per_token.sum())
    torch.testing.assert_close(stats[1],per_token.sum())
    torch.testing.assert_close(logits.grad,expected,rtol=1e-5,atol=1e-6)


def test_exact_kl_zero_and_zero_gradient_at_reference():
    logits=torch.tensor([[1.,-2.,.3]],requires_grad=True)
    reference=(logits.detach()/.6).log_softmax(-1)
    old=reference[:,0]
    loss,stats=token_objective(logits,torch.tensor([0]),reference,old,0.,entropy_coefficient=0.)
    loss.backward()
    assert abs(stats[1].item())<1e-6
    assert logits.grad.norm().item()<1e-6


def test_exact_kl_avoids_sampled_exponential_outlier():
    logits=torch.tensor([[-14.,0.,-1.]],requires_grad=True)
    labels=torch.tensor([0]);reference=(torch.tensor([[-.1,0.,-1.]])/.6).log_softmax(-1)
    old=torch.tensor([-.2])
    exact,stats=token_objective(logits,labels,reference,old,0.,entropy_coefficient=0.)
    sampled,_=token_objective(logits,labels,reference[:,0],old,0.,entropy_coefficient=0.)
    exact.backward()
    assert torch.isfinite(exact) and torch.isfinite(logits.grad).all()
    assert sampled.item()>1000*exact.item()
    assert logits.grad.norm().item()<.1


def test_exact_kl_validates_vocabulary_shape():
    with pytest.raises(ValueError,match='same token vocabulary'):
        token_objective(torch.zeros(2,3),torch.tensor([0,1]),torch.zeros(2,2),torch.zeros(2),0.)


def test_scheduled_kl_migration_is_explicit_and_cannot_relabel_past_updates():
    from dataclasses import replace
    from .hero_v2 import identity,kl_estimator,validate_resume_identity
    from .test_hero_v2 import config
    old=config()
    new=replace(old,source_sha256='b'*64,exact_kl_start_update=215,
        accepted_resume_sources=('a'*64,),accepted_kl_schedule_sources=('a'*64,))
    assert kl_estimator(new,214)=='k3' and kl_estimator(new,215)=='categorical'
    validate_resume_identity(identity(old),new,policy_version=210)
    for version in (None,215,220):
        with pytest.raises(ValueError,match='Migration cannot'):
            validate_resume_identity(identity(old),new,policy_version=version)
    with pytest.raises(ValueError,match='Migration cannot'):
        validate_resume_identity(identity(old),replace(new,accepted_kl_schedule_sources=()),policy_version=210)
    with pytest.raises(ValueError,match='Migration cannot'):
        validate_resume_identity(identity(old),replace(new,learning_rate=1e-5),policy_version=210)
    validate_resume_identity(identity(new),new,policy_version=220)
