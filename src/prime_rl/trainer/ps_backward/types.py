from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Trace:
    """One prompt and completion; positions survive every crop and replay."""

    input_ids: torch.Tensor
    prefix_length: int
    position_ids: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    old_logps: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    provenance: dict = field(default_factory=dict)
    temperatures: torch.Tensor | None = None

    def __post_init__(self):
        if self.input_ids.ndim != 1 or self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a one-dimensional int64 tensor")
        if not 1 <= self.prefix_length < self.input_ids.numel():
            raise ValueError("A trace needs a nonempty prompt and completion")
        if self.position_ids is not None:
            if self.position_ids.shape != self.input_ids.shape or self.position_ids.dtype != torch.long:
                raise ValueError("position_ids must be int64 and match input_ids")
        for name in ("advantages", "old_logps", "loss_mask", "temperatures"):
            value = getattr(self, name)
            if value is not None and value.shape != (self.completion_length,):
                raise ValueError(f"{name} must have one entry per completion token")
            if value is not None and value.requires_grad:
                raise ValueError(f"{name} must be detached")
        if self.loss_mask is not None and self.loss_mask.dtype != torch.bool:
            raise ValueError("loss_mask must be boolean")
        if self.temperatures is not None and (
            not torch.isfinite(self.temperatures).all() or not (self.temperatures > 0).all()
        ):
            raise ValueError("temperatures must be finite and positive")

    @property
    def completion_length(self):
        return self.input_ids.numel() - self.prefix_length

    @property
    def positions(self):
        if self.position_ids is not None:
            return self.position_ids
        return torch.arange(self.input_ids.numel(), device=self.input_ids.device)

    def to(self, device):
        fields = {}
        for name in ("input_ids", "position_ids", "advantages", "old_logps", "loss_mask", "temperatures"):
            value = getattr(self, name)
            fields[name] = value.to(device) if value is not None else None
        return Trace(prefix_length=self.prefix_length, provenance=self.provenance, **fields)


@dataclass
class BackwardResult:
    loss: float
    loss_tokens: int
