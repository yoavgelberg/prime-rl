import argparse
import json
import os
from pathlib import Path
import time

import torch

from .decode_fa4 import full_decode


@torch.inference_mode()
def check():
    torch.manual_seed(91)
    conditions=[]
    for batch,length,position in ((8,2048,177),(8,8192,8191),(8,32768,32750)):
        q=torch.randn(batch,1,32,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)
        k=torch.randn(batch,length,8,128,device='cuda',dtype=torch.bfloat16).transpose(1,2)
        v=torch.randn_like(k)
        positions=torch.full((batch,1),position,device='cuda',dtype=torch.long)
        outputs,timings={},{}
        for splits in (1,0):
            os.environ['PS_CONCENTRATION_FULL_SPLITS']=str(splits)
            started=time.perf_counter()
            for _ in range(3): full_decode(q,k,v,positions)
            torch.cuda.synchronize()
            cold=time.perf_counter()-started
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output=full_decode(q,k,v,positions)
            for _ in range(5): graph.replay()
            start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(50): graph.replay()
            end.record();end.synchronize()
            outputs[splits]=output.clone()
            timings[splits]=dict(cold_setup_seconds=cold,cached_ms=start.elapsed_time(end)/50)
            del graph,output
        relative=float((outputs[1].float()-outputs[0].float()).norm()/outputs[1].float().norm())
        assert relative<.01,(batch,length,position,relative)
        conditions.append(dict(batch=batch,keys=length,position=position,relative_l2=relative,
                               timings=timings,speedup=timings[1]['cached_ms']/timings[0]['cached_ms']))
        del q,k,v,outputs
    return dict(status='GO',scope='Existing FA4 split heuristic versus one split, including fully masked trailing partitions',
                conditions=conditions,gpu=torch.cuda.get_device_name(),generated_token_fidelity_pending=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=check()
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
