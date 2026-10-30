"""Reference k3 KL and rollout/recomputed-policy audits on complete native games."""

import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from .activation_offload import activation_offload
from .check_policy import accept_parity, teacher_logps
from .objective import sampled_kl
from .policy import load_model


def accumulate_kl(policy, records, *, coefficient, outcome_config, reference_checkpoint,
                  output, initial_policy=False, temperature=.6, allow_numerical_target_miss=False):
    reference_checkpoint = Path(reference_checkpoint)
    if reference_checkpoint.name != '768f209d9ea81521153ed38c47d515654e938aea':
        raise ValueError('The KL reference must be the frozen starting snapshot')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reference = None if initial_policy else load_model(str(reference_checkpoint), str(policy.device))
    if reference is not None:
        reference.requires_grad_(False)
    differences, tokens, kl_sum = [], 0, 0.
    try:
        for row in records:
            with gzip.open(row['token_record'], 'rt') as handle:
                record = json.load(handle)
            selected = record['loss_mask']
            if sum(selected) != row['generated_tokens'] or record['prefix_length'] > 512:
                raise ValueError('Invalid sampled-token accounting or system-prefix length')
            old = torch.tensor([value for value, keep in zip(record['old_tempered_logps'], selected, strict=True) if keep],
                device=policy.device)
            if initial_policy:
                with torch.no_grad():
                    logps = teacher_logps(policy.base, record, temperature=temperature, native_vllm=True)
                local_kl = 0.
            else:
                with torch.no_grad():
                    ref = teacher_logps(reference, record, temperature=temperature, native_vllm=True)
                with activation_offload(policy.parameters()):
                    logps = teacher_logps(policy.base, record, grad=True, temperature=temperature, native_vllm=True)
                    values = sampled_kl(logps, ref)
                    loss = coefficient * values.sum() / (outcome_config.denominator * outcome_config.global_batch_size)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Reference KL became nonfinite')
                local_kl = float(values.detach().sum())
                loss.backward()
                del ref, values, loss
            if len(logps) != sum(selected) or len(logps) != len(old):
                raise ValueError('KL must cover every sampled thinking/action token exactly once')
            differences.append((logps.detach() - old).abs().cpu().numpy())
            kl_sum += local_kl
            tokens += len(logps)
            del logps, old
            print(json.dumps(dict(event='reference_kl_trajectory', episode_id=row['episode_id'],
                sampled_tokens=row['generated_tokens'], initial_policy=initial_policy)), flush=True)
    finally:
        del reference
        torch.cuda.empty_cache()
    differences = np.concatenate(differences)
    audit = dict(mean_abs=float(differences.mean()), p99=float(np.quantile(differences, .99)),
        max_abs=float(differences.max()), mean_limit=.01, p99_limit=.05,
        generated_tokens=tokens, direct_loss_scope='sampled assistant only; forced tokens excluded',
        initial_policy=initial_policy, exploratory_target_miss_allowed=allow_numerical_target_miss,
        arithmetic='Native effective BF16 projections/embeddings/norm weights, FP32 pre-round fused residual sums, BF16 RoPE cache')
    audit['status'] = 'GO' if audit['mean_abs'] <= .01 and audit['p99'] <= .05 else 'HOLD'
    (output / 'rollout-teacher-parity.json').write_text(json.dumps(audit, indent=2) + '\n')
    if not accept_parity(audit, allow_numerical_target_miss):
        raise RuntimeError(f'Recomputed policy likelihood audit failed before optimizer update: {audit}')
    return dict(kl_sum=kl_sum, generated_tokens=tokens,
        normalized_kl=kl_sum / (outcome_config.denominator * outcome_config.global_batch_size),
        kl_per_generated_token=kl_sum / tokens, reference_kl_seconds=time.perf_counter() - started,
        initial_kl_zero_by_identical_policy=initial_policy, parity=audit)
