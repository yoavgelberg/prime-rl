"""Update recorded-backward compute weights without invalidating captured pointers."""

import torch


@torch.no_grad()
def prepare_weights(model, parameters, dtype):
    weights = {}
    for parameter in parameters.values():
        if parameter.ndim != 2:
            continue
        stamp = parameter._version, parameter.data_ptr(), getattr(model, '_ps_update_version', 0)
        cached = model._weight_cast_cache.get(id(parameter))
        if cached is None:
            value = parameter.detach().to(dtype)
        else:
            value = cached[1]
            if value.shape != parameter.shape or value.dtype != dtype or value.device != parameter.device:
                raise ValueError('Recorded compute-weight layout changed after graph construction')
            if cached[0] != stamp:
                value.copy_(parameter)
        model._weight_cast_cache[id(parameter)] = stamp, value
        weights[id(parameter)] = value
    native_dtype = getattr(model, 'recorded_native_weight_dtype', None)
    if native_dtype is not None:
        for module in model.base.modules():
            if not hasattr(module, 'variance_epsilon'):
                continue
            parameter = module.weight
            value = getattr(module, '_ps_recorded_compute_weight', None)
            if value is None:
                value = parameter.detach().to(native_dtype).to(parameter.dtype)
                module._ps_recorded_compute_weight = value
            else:
                value.copy_(parameter.to(native_dtype))
    return weights
