from types import SimpleNamespace

import torch

from prime_rl.trainer.ps_backward.recorded import recorded_rms
from prime_rl.trainer.ps_backward.recorded_qkv import prepare_qkv_weights
from prime_rl.trainer.ps_backward.recorded_weights import prepare_weights


def toy_model():
    torch.manual_seed(31)
    base = torch.nn.Module()
    base.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    for name, size in (('q_proj', 4), ('k_proj', 2), ('v_proj', 2)):
        setattr(layer.self_attn, name, torch.nn.Linear(4, size, bias=False))
    base.model.layers = torch.nn.ModuleList([layer])
    base.model.norm = torch.nn.RMSNorm(4, eps=1e-6)
    base.model.norm.variance_epsilon = 1e-6
    with torch.no_grad():
        base.model.norm.weight.copy_(torch.tensor([.7317, 1.2751, .9913, .3459]))
    base.lm_head = torch.nn.Linear(4, 9, bias=False)
    model = SimpleNamespace(base=base, _weight_cast_cache={}, recorded_native_weight_dtype=torch.bfloat16)
    return model, dict(base.named_parameters())


def test_real_adam_updates_refresh_compute_weights_without_replacing_graph_pointers():
    model, parameters = toy_model()
    weights = prepare_weights(model, parameters, torch.bfloat16)
    prepare_qkv_weights(model, weights)
    pointers = {key: value.data_ptr() for key, value in weights.items()}
    packed_pointer = model._recorded_qkv_weights[0][1].data_ptr()
    norm = model.base.model.norm
    norm_pointer = norm._ps_recorded_compute_weight.data_ptr()
    old = {key: value.clone() for key, value in weights.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=.03, weight_decay=0)
    for parameter in parameters.values():
        parameter.grad = torch.full_like(parameter, .1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated = prepare_weights(model, parameters, torch.bfloat16)
    prepare_qkv_weights(model, updated)
    assert pointers == {key: value.data_ptr() for key, value in updated.items()}
    assert model._recorded_qkv_weights[0][1].data_ptr() == packed_pointer
    assert norm._ps_recorded_compute_weight.data_ptr() == norm_pointer
    for parameter in parameters.values():
        if parameter.ndim == 2:
            torch.testing.assert_close(updated[id(parameter)], parameter.detach().bfloat16(), rtol=0, atol=0)
    assert any(not torch.equal(old[key], value) for key, value in updated.items())
    projections = model.base.model.layers[0].self_attn
    expected = torch.cat([getattr(projections, name).weight.detach().bfloat16()
                          for name in ('q_proj', 'k_proj', 'v_proj')])
    torch.testing.assert_close(model._recorded_qkv_weights[0][1], expected, rtol=0, atol=0)
    torch.testing.assert_close(norm._ps_recorded_compute_weight, norm.weight.detach().bfloat16().float(), rtol=0, atol=0)


def test_norm_input_gradient_uses_effective_native_weight_and_master_gradient_is_fp32():
    model, parameters = toy_model()
    prepare_weights(model, parameters, torch.bfloat16)
    norm = model.base.model.norm
    x = torch.randn(7, 4, requires_grad=True)
    seed = torch.randn_like(x)
    effective = norm.weight + (norm.weight.detach().bfloat16().float() - norm.weight).detach()
    expected = x * (x.square().mean(-1, keepdim=True) + norm.variance_epsilon).rsqrt() * effective
    reference = torch.autograd.grad(expected, (x, norm.weight), seed)
    actual = recorded_rms(x, norm, expected.detach())
    gradients = torch.autograd.grad(actual, (x, norm.weight), seed)
    for result, target in zip(gradients, reference, strict=True):
        torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-6)
    assert gradients[1].dtype == torch.float32
