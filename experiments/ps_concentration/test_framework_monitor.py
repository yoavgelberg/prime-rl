from types import SimpleNamespace

import torch


def test_configured_grpo_dispatches_through_actual_framework():
    from prime_rl.orchestrator.advantage import compute_advantages
    from prime_rl.orchestrator.config import CustomAdvantageConfig
    from prime_rl.orchestrator.grpo import normalized_advantages

    config = CustomAdvantageConfig(import_path='prime_rl.orchestrator.grpo.grpo_advantage_fn', kwargs={'epsilon':1e-4})
    rewards = [0.,1.,2.,3.,4.,5.,6.,7., *([.5] * 8)]
    actual = compute_advantages(rewards, [11] * 16, 8, config)
    expected = normalized_advantages(torch.tensor(rewards).reshape(2, 8)).reshape(-1)
    torch.testing.assert_close(torch.tensor(actual), expected, rtol=0, atol=0)


def test_standard_monitor_passes_config_and_rl_metrics_to_wandb(monkeypatch, tmp_path):
    import wandb
    from prime_rl.utils.config import WandbWithExtrasConfig
    import prime_rl.utils.monitor as module

    from prime_rl.utils import logger as logger_module
    monkeypatch.setattr(logger_module, '_LOGGER', None)
    logger_module.setup_logger("INFO")
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setattr(module, '_MONITOR', None)
    initialized, logged = [], []
    monkeypatch.setattr(wandb, 'init', lambda **kwargs: initialized.append(kwargs) or SimpleNamespace(id='unit-only'))
    monkeypatch.setattr(wandb, 'log', lambda metrics, step=None: logged.append((metrics, step)))
    config = WandbWithExtrasConfig(project='unit-only', name='unit-only', id='unit-only', offline=True, log_extras=None)
    monitor = module.setup_monitor(wandb_config=config, output_dir=tmp_path)
    metrics = dict(policy_loss=.1, kl_per_generated_token=.02, entropy=.5, reward_mean=.25,
        advantage_std=.9, gradient_norm=.7, learning_rate=5e-6, pairs_found=4., generated_tokens=10000)
    monitor.log(metrics, step=1)
    assert initialized[0]['project'] == 'unit-only'
    assert initialized[0]['resume'] == 'allow'
    assert initialized[0]['mode'] == 'offline'
    assert logged == [(metrics, 1)]
    assert monitor.history == [metrics]
