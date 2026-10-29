import torch
from torch import nn
import torch.nn.functional as F

from .checkpoints import inference_state


def test_compact_checkpoint_preserves_effective_policy_and_direct_fp32_values():
    torch.manual_seed(51)
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=nn.Embedding(31,8)
            self.norm=nn.LayerNorm(8)
            self.projection=nn.Linear(8,31)
        def forward(self,ids):
            hidden=self.norm(self.embedding(ids))
            return F.linear(hidden.bfloat16(),self.projection.weight.bfloat16(),None).float()+self.projection.bias
    policy=Policy()
    with torch.no_grad():policy.norm.weight.add_(.00031)
    restored=Policy()
    compact=inference_state(policy)
    assert compact['embedding.weight'].dtype==torch.float32
    assert compact['norm.weight'].dtype==torch.float32
    assert compact['projection.bias'].dtype==torch.float32
    assert compact['projection.weight'].dtype==torch.bfloat16
    restored.load_state_dict(compact)
    ids=torch.tensor([[1,2,3],[17,9,4]])
    torch.testing.assert_close(policy(ids),restored(ids),rtol=0,atol=0)
