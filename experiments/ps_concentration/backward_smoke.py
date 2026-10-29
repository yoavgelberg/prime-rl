import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from transformers import AutoTokenizer

from .audit import token_audit
from .check_policy import accept_parity, parity, teacher_logps
from .checkpoints import save_checkpoint
from .config import Config, generation_seed
from .objective import leave_one_out, trajectory_loss
from .policy import load_model, StreamingQwen
from .rollout import run_batch, save_episodes
from .resume_check import check as check_resume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--attention', choices=['ps','full'], default='ps')
    parser.add_argument('--allow-numerical-target-miss', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    optimizer_resume=check_resume()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = Config(model=args.model, attention=args.attention, transcript_cap=token_audit(tokenizer)['52']['required_cap'])
    policy = load_model(args.model)
    reference = load_model(args.model)
    records = run_batch(policy, tokenizer, [1234]*8, [generation_seed(11, 0, 0, i) for i in range(8)], config)
    save_episodes(output/'episodes.jsonl.gz', records)
    audit = parity(policy, records[:1])
    if not accept_parity(audit, args.allow_numerical_target_miss):
        (output/'parity.json').write_text(json.dumps(audit, indent=2))
        raise RuntimeError(f'Fresh sampled trajectory parity failed: {audit}')
    policy.requires_grad_(True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-6/8, betas=(.9,.95), eps=1e-8, weight_decay=0, foreach=True)
    advantages = leave_one_out([[r['native_return'] for r in records]])[0]
    started = time.perf_counter()
    losses, tokens = [], 0
    for record, advantage in zip(records, advantages, strict=True):
        with torch.no_grad():
            ref = teacher_logps(reference, record, temperature=.8)
        logps = teacher_logps(policy, record, grad=True, temperature=.8)
        assert len(logps) == sum(record['loss_mask'])
        loss = trajectory_loss(logps, ref, advantage, config, global_trajectories=8)
        assert torch.isfinite(loss)
        loss.backward()
        losses.append(float(loss.detach()))
        tokens += len(logps)
        del logps, ref, loss
    parameters = list(policy.parameters())
    assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in parameters)
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
    optimizer.step()
    delta = sum(float((a.detach()-b).square().sum()) for a,b in zip(policy.parameters(),reference.parameters(),strict=True))**.5
    update_seconds=time.perf_counter()-started
    checkpoint=output/'reload-checkpoint'
    save_checkpoint(checkpoint,policy,tokenizer,dict(scope='Discarded system qualification'))
    restored=load_model(str(checkpoint))
    original_stream=StreamingQwen(policy,1,4,16,'ps',128)
    restored_stream=StreamingQwen(restored,1,4,16,'ps',128)
    ids=torch.tensor([codec for codec in range(32)],device=policy.device)
    maximum=0.
    for token in ids:
        before=original_stream.step(token[None])
        after=restored_stream.step(token[None])
        maximum=max(maximum,float((before-after).abs().max()))
        torch.testing.assert_close(before,after,rtol=0,atol=0)
    result = dict(status='GO', scope='Systems smoke with eight on-policy trajectories; discarded optimizer and weights; not a scientific pilot',
                  optimizer_resume=optimizer_resume,
                  attention=args.attention, parity=audit, returns=[r['native_return'] for r in records],
                  advantages=advantages.tolist(), losses=losses, sampled_loss_tokens=tokens,
                  gradient_norm=norm, parameter_update_l2=delta, backward_optimizer_seconds=update_seconds,
                  checkpoint_reload_max_logit_error=maximum,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  nonzero_task_signal=bool(np.any(advantages != 0)))
    result['exploratory_target_miss_allowed'] = args.allow_numerical_target_miss
    (output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
