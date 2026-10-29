import numpy as np


def leave_one_out(returns):
    returns = np.asarray(returns, dtype=np.float64)
    assert returns.ndim == 2 and returns.shape[1] == 8
    advantages=returns-(returns.sum(-1,keepdims=True)-returns)/7
    advantages[np.ptp(returns,axis=1)==0]=0.
    return advantages


def sampled_kl(logps, reference):
    ratio = reference - logps
    return ratio.exp() - ratio - 1


def trajectory_loss(logps, reference, advantage, config, global_trajectories=64):
    return (-float(advantage)*logps + config.kl_coefficient*sampled_kl(logps, reference)).sum() / (config.denominator*global_trajectories)
