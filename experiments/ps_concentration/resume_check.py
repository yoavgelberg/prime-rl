import argparse
import io
import json
from pathlib import Path

import torch

from .checkpoints import restore_adamw


def check():
    before_cpu,before_cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
    torch.manual_seed(911)
    parameters=[torch.nn.Parameter(torch.randn(shape,device='cuda')) for shape in ((11,7),(5,))]
    optimizer=torch.optim.AdamW(parameters,lr=5e-6/8,betas=(.9,.95),eps=1e-8,weight_decay=0,foreach=True)
    for p in parameters: p.grad=torch.randn_like(p)
    torch.nn.utils.clip_grad_norm_(parameters,1.)
    optimizer.step()
    buffer=io.BytesIO()
    torch.save(dict(masters=[p.detach().clone() for p in parameters],optimizer=optimizer.state_dict(),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),buffer)
    expected_cpu=torch.randn(5)
    for p in parameters: p.grad=torch.randn_like(p)
    torch.nn.utils.clip_grad_norm_(parameters,1.)
    optimizer.param_groups[0]['lr']=5e-6*2/8
    optimizer.step()
    buffer.seek(0)
    saved=torch.load(buffer,map_location='cuda',weights_only=False)
    restored=[torch.nn.Parameter(p) for p in saved['masters']]
    resumed=torch.optim.AdamW(restored,lr=5e-6,betas=(.9,.95),eps=1e-8,weight_decay=0,foreach=True)
    restore_adamw(resumed,saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng'].cpu())
    torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    torch.testing.assert_close(torch.randn(5),expected_cpu,rtol=0,atol=0)
    for p in restored: p.grad=torch.randn_like(p)
    torch.nn.utils.clip_grad_norm_(restored,1.)
    resumed.param_groups[0]['lr']=5e-6*2/8
    resumed.step()
    for expected,actual in zip(parameters,restored,strict=True):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(resumed.state[actual][key],optimizer.state[expected][key],rtol=0,atol=0)
        assert resumed.state[actual]['step'].device.type=='cpu'
    torch.set_rng_state(before_cpu)
    torch.cuda.set_rng_state(before_cuda)
    return dict(status='GO',scope='Two AdamW updates versus interrupted restore with CUDA-mapped checkpoint',
                exact_parameters=True,exact_moments=True,exact_cpu_cuda_rng=True,step_scalars_on_cpu=True,
                learning_rate=5e-6*2/8)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=check()
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
