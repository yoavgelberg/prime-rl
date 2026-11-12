"""Reference k3 KL and rollout/recomputed-policy audits on complete native games."""

import gzip
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .activation_offload import activation_offload
from .check_policy import teacher_logps
from .objective import sampled_kl
from .policy import load_model


def zero_kl_outliers(audit):
    rows = audit.get('teacher_outliers', [])
    return (bool(rows) and len(rows) == audit.get('teacher_outlier_count')
            and all(math.isfinite(row['policy_logp']) and row['policy_logp'] == row['reference_logp']
                    for row in rows))


def teacher_outliers(logps, ref, old, record):
    indices = ((logps.detach() - old).abs() > 2.).nonzero().flatten().tolist()
    positions = [index for index, keep in enumerate(record['loss_mask']) if keep] if indices else []
    return [dict(episode_id=record['episode_id'], position=positions[index],
        token_id=record['tokens'][positions[index]], policy_logp=float(logps[index].detach()),
        reference_logp=float(ref[index]), rollout_logp=float(old[index])) for index in indices]


def accept_training_audit(audit, records, exploratory):
    native_errors = [row.get('native_capture_mean_loss_error', float('inf')) for row in records]
    if not native_errors or not all(error <= 1e-6 for error in native_errors):
        return False
    if audit['status'] == 'GO':
        return True
    return (exploratory and audit['mean_abs'] <= .03 and audit['p99'] <= .35
            and math.isfinite(audit['max_abs'])
            and (audit['max_abs'] <= 2. or audit.get('initial_policy') is True or zero_kl_outliers(audit)))


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
    differences, tokens, kl_sum, outliers = [], 0, 0., []
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
                outliers.extend(teacher_outliers(logps, ref, old, record))
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
    audit['native_capture_max_mean_loss_error'] = max(row.get('native_capture_mean_loss_error', float('inf')) for row in records)
    audit['exploratory_limits'] = dict(mean_abs=.03, p99=.35, max_abs=2., native_capture_mean_loss_error=1e-6)
    audit['initial_zero_kl_outlier_exception'] = initial_policy and audit['max_abs'] > 2.
    audit['teacher_outlier_count'] = int(np.count_nonzero(differences > 2.))
    audit['teacher_outliers'] = outliers
    audit['zero_kl_outlier_exception'] = zero_kl_outliers(audit)
    audit['teacher_kl_approximate'] = audit['status'] != 'GO'
    audit['training_runtime_accepted'] = accept_training_audit(audit, records, allow_numerical_target_miss)
    (output / 'rollout-teacher-parity.json').write_text(json.dumps(audit, indent=2) + '\n')
    if not audit['training_runtime_accepted']:
        raise RuntimeError(f'Recomputed policy likelihood audit failed before optimizer update: {audit}')
    print(json.dumps(dict(event='reference_kl_audit', **audit)), flush=True)
    return dict(kl_sum=kl_sum, generated_tokens=tokens,
        normalized_kl=kl_sum / (outcome_config.denominator * outcome_config.global_batch_size),
        kl_per_generated_token=kl_sum / tokens, reference_kl_seconds=time.perf_counter() - started,
        initial_kl_zero_by_identical_policy=initial_policy, parity=audit)
