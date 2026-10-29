import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prime_rl.trainer.ps_backward.attention import dense_ps_attention
from .linear import ConcentrationQwen

from .config import Config
from .metrics import read_episodes
from .policy import load_model


def toy_reference():
    torch.manual_seed(17)
    indices = torch.arange(128, device='cuda')
    q = torch.randn(1, 4, 128, 32, device='cuda', requires_grad=True)
    k = torch.randn(1, 2, 128, 32, device='cuda', requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = dense_ps_attention(q, k, v, indices, indices, 4, 16)
    seed = torch.randn_like(actual)
    actual.backward(seed)
    grads = [x.grad.clone() for x in (q, k, v)]
    qr, kr, vr = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    mask = (indices[None] <= indices[:, None]) & ((indices[None] < 4) | (indices[None] >= indices[:, None]-15))
    scores = qr @ kr.repeat_interleave(2, 1).transpose(-1, -2) / 32**.5
    reference = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vr.repeat_interleave(2, 1)
    reference.backward(seed)
    errors = [float((a-b).norm()/b.norm()) for a, b in zip(
        [actual, *grads], [reference, qr.grad, kr.grad, vr.grad], strict=True)]
    assert max(errors) < 1e-4, errors
    return dict(prefix=4, window=16, tokens=128, fp32_output_gradient_relative_l2=errors,
                limit=1e-4, scope='FP32 dense PS adapter versus independent explicit masked softmax')


@torch.autocast('cuda', dtype=torch.bfloat16)
def teacher_logps(base, record, backend='fa4', *, grad=False, temperature=1.):
    device = base.device
    ids = torch.tensor(record['tokens'], device=device)
    padding = (-(len(ids)-1)) % 1024
    if padding:
        ids = torch.nn.functional.pad(ids, (0, padding), value=int(ids[-1]))
    indices = torch.arange(len(ids)-1, device=device)
    window = record['window'] if record['attention'] == 'ps' else len(ids) + 1
    adapter = ConcentrationQwen(base, window, backend=backend, checkpoint_layers=grad)
    hidden = adapter.segment(ids[:-1], indices, indices, record['prefix_length'])
    selected = torch.tensor(record['loss_mask'][1:] + [False]*padding, device=device)
    labels = ids[1:][selected]
    temperatures = torch.full((len(labels),), temperature, device=device)
    return adapter.logps(hidden[selected], labels, block_size=128, temperatures=temperatures)


@torch.no_grad()
def parity(base, records, backend='fa4'):
    differences = []
    counts = []
    for record in records:
        actual = teacher_logps(base, record, backend)
        expected = torch.tensor([p for p, mask in zip(record['old_logps'], record['loss_mask'], strict=True) if mask],
                                device=base.device)
        differences.extend((actual-expected).abs().cpu().tolist())
        counts.append(dict(sampled_tokens=sum(record['loss_mask']),
                           early_sampled_tokens=sum(record['loss_mask'][:record['prefix_length']+record['window']]),
                           eviction_crossed=len(record['tokens']) > record['prefix_length'] + record['window']))
        assert len(actual) == sum(record['loss_mask'])
    result = dict(mean_abs=float(np.mean(differences)), p99=float(np.quantile(differences, .99)),
                  max_abs=float(max(differences)), mean_limit=.01, p99_limit=.05, token_accounting=counts,
                  backend=backend)
    result['status'] = 'GO' if result['mean_abs'] <= .01 and result['p99'] <= .05 else 'HOLD'
    return result


def accept_parity(audit, exploratory=False):
    if audit['status'] == 'GO':
        return True
    return exploratory and audit['mean_abs'] <= .02 and audit['p99'] <= .15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', default='fa4', choices=['fa4', 'dense'])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_model(args.model)
    result = dict(toy=toy_reference(), parity=parity(base, read_episodes(args.episodes)[:2], args.backend))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if result['parity']['status'] != 'GO':
        raise RuntimeError('Rollout/training parity failed; investigate before training')


if __name__ == '__main__':
    main()
