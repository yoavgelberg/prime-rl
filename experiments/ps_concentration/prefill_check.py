import argparse
import json
from pathlib import Path
import time

import torch

from .chunk_prefill import append_tokens
from .policy import load_model, StreamingQwen


@torch.inference_mode()
def check(base):
    results={}
    for name,attention,prefix,window,beginning,block in (
            ('ps-small','ps',4,16,20,8),('ps-native','ps',152,1024,1176,32),
            ('full','full',152,1024,2020,32)):
        eager=StreamingQwen(base,2,prefix,window,attention,4096)
        chunked=StreamingQwen(base,2,prefix,window,attention,4096)
        torch.manual_seed(91)
        tokens=torch.randint(0,base.config.vocab_size,(2,beginning+4*(block+1)),device=base.device)
        eager.prefill(tokens[:,:beginning])
        chunked.prefill(tokens[:,:beginning])
        pinned=[k[:,:,:prefix].clone() for k in chunked.keys]
        cursor=beginning
        errors,prob_errors,times=[],[],[]
        for iteration in range(4):
            torch.cuda.synchronize()
            started=time.perf_counter()
            for index in range(cursor,cursor+block):
                eager.step(tokens[:,index])
            torch.cuda.synchronize()
            sequential_seconds=time.perf_counter()-started
            started=time.perf_counter()
            append_tokens(chunked,tokens[:,cursor:cursor+block])
            torch.cuda.synchronize()
            chunk_seconds=time.perf_counter()-started
            cursor+=block
            expected=eager.step(tokens[:,cursor])
            actual=chunked.step(tokens[:,cursor])
            cursor+=1
            relative=float((expected-actual).norm()/expected.norm())
            errors.append(relative)
            labels=expected.argmax(-1,keepdim=True)
            prob_errors.extend((expected.log_softmax(-1).gather(1,labels)-actual.log_softmax(-1).gather(1,labels)).abs().flatten().tolist())
            times.append(dict(iteration=iteration,queued_tokens=block,sequential_seconds=sequential_seconds,
                              chunk_seconds=chunk_seconds,cold_compile_included=iteration==0))
        cache_errors=[float((a.float()-b.float()).norm()/a.float().norm())
                      for a,b in zip(eager.keys,chunked.keys,strict=True)]
        for old,new in zip(pinned,chunked.keys,strict=True):
            torch.testing.assert_close(new[:,:,:prefix],old,rtol=0,atol=0)
        results[name]=dict(max_relative_logit_l2=max(errors),max_relative_key_l2=max(cache_errors),
                           within_probe_tolerance=max(errors)<.01 and max(cache_errors)<.02,
                           top1_logp_mean_abs=sum(prob_errors)/len(prob_errors),top1_logp_max_abs=max(prob_errors),
                           pinned_prefix_exact=True,cache_tokens=chunked.capacity,timings=times)
        del eager,chunked
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--collect-only',action='store_true')
    args=parser.parse_args()
    base=load_model(args.model)
    conditions=check(base)
    result=dict(status='GO' if all(v['within_probe_tolerance'] for v in conditions.values()) else 'HOLD',
                scope='Queued-token append numerical and eviction checks; fresh rollout parity still required',
                conditions=conditions,gpu=torch.cuda.get_device_name(),logit_l2_limit=.01,key_l2_limit=.02)
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='GO' and not args.collect_only:
        raise RuntimeError('Probe tolerance missed; all conditions and timings preserved in receipt')


if __name__=='__main__':
    main()
