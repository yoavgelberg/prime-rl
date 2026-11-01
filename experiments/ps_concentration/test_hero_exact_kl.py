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
