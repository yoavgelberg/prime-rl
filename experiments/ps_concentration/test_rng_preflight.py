from .rng_preflight import accepted


def test_preflight_requires_maximum_length_and_unchanged_weights():
    source = 'a' * 64
    result = dict(source_sha256=source, raw_tokens=262144, learned_parameter_updates=0,
        zero_gradient_optimizer_weights_unchanged=True, early_loss_tokens=926,
        optimizer_states_resident=True, mean_abs=.006, p99=.065)
    assert accepted(result, source)
    for field, value in [('raw_tokens', 202085), ('source_sha256', 'b' * 64),
                          ('learned_parameter_updates', 1), ('zero_gradient_optimizer_weights_unchanged', False),
                          ('mean_abs', .03), ('p99', .16)]:
        changed = dict(result, **{field: value})
        assert not accepted(changed, source)
