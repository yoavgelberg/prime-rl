import pytest
import torch

from .rng_overlap_loss import accept_training_audit
from prime_rl.trainer.ps_backward.checkpoint import save_pending_gradients


def test_exploratory_teacher_drift_requires_native_captured_policy_certification():
    audit = dict(status='HOLD', mean_abs=.0166, p99=.256, max_abs=1.21)
    rows = [dict(native_capture_mean_loss_error=6.4e-8)]
    assert accept_training_audit(audit, rows, True)
    assert not accept_training_audit(audit, rows, False)
    assert not accept_training_audit(audit, [{}], True)
    assert not accept_training_audit(audit, [], True)
    assert not accept_training_audit(audit, [dict(native_capture_mean_loss_error=1e-3)], True)
    for name, value in [('mean_abs', .04), ('p99', .5), ('max_abs', 3.), ('mean_abs', float('nan'))]:
        assert not accept_training_audit(dict(audit, **{name:value}), rows, True)


def test_pending_task_gradient_save_preserves_values_and_does_not_modify_live_gradient(tmp_path):
    model = torch.nn.Linear(3, 2)
    model(torch.ones(2, 3)).sum().backward()
    before = {name:parameter.grad.clone() for name,parameter in model.named_parameters()}
    records = [dict(episode_id='u0001-g0-r0', native_capture_mean_loss_error=1e-8)]
    path = tmp_path / 'pending/rank-0.pt'
    receipt = save_pending_gradients(path, model, records, policy_version=0, metadata=dict(seed=11))
    saved = torch.load(path, weights_only=True)
    assert receipt['episodes'] == 1 and saved['policy_version'] == 0
    assert saved['records'] == records and saved['metadata'] == dict(seed=11)
    for name, parameter in model.named_parameters():
        assert torch.equal(saved['gradients'][name], before[name])
        assert torch.equal(parameter.grad, before[name])
    model.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match='cover all trainable'):
        save_pending_gradients(tmp_path / 'bad.pt', model, records, policy_version=0, metadata={})


def test_initial_identical_reference_has_zero_kl_and_reports_rare_teacher_outlier():
    audit = dict(status='HOLD', mean_abs=.01795, p99=.271, max_abs=2.187, initial_policy=True)
    rows = [dict(native_capture_mean_loss_error=5.3e-8)]
    assert accept_training_audit(audit, rows, True)
    assert not accept_training_audit(dict(audit,initial_policy=False),rows,True)
    assert not accept_training_audit(audit,rows,False)
    assert not accept_training_audit(dict(audit,mean_abs=.46),rows,True)
    assert not accept_training_audit(dict(audit,max_abs=float('inf')),rows,True)
