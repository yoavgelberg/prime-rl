"""Reconstruct redundant pointwise fields of a compact inference tape."""

import torch


@torch.no_grad()
def restore_values(record):
    for name, raw, normalized in (("input", "norm_input", "norm1"), ("mid", "norm_mid", "norm2")):
        if name not in record and raw in record and normalized in record:
            record[name] = record[raw].to(record[normalized].dtype)
    if "gated" not in record and "gate_up" in record:
        from .recording_kernels import silu_value

        record["gated"] = silu_value(record["gate_up"])
    return record
