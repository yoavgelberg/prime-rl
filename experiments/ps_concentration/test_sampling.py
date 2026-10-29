import numpy as np
import torch

from .rollout import sample


class LargestUniform:
    def random(self):
        return np.nextafter(1.,0.)


def test_top_p_cdf_rounding_cannot_select_a_zero_mass_tail():
    torch.manual_seed(91)
    logits=torch.randn(64,127)*3
    tokens,raw,tempered=sample(logits,[LargestUniform() for _ in range(64)],.8,.95)
    probabilities=(logits.double()/.8).softmax(-1)
    values,indices=probabilities.sort(-1,descending=True)
    for row,token in enumerate(tokens):
        retained=indices[row][values[row].cumsum(-1)-values[row]<.95]
        assert token in retained.tolist()
    targets=torch.tensor(tokens)[:,None]
    torch.testing.assert_close(torch.tensor(raw),logits.log_softmax(-1).gather(1,targets)[:,0])
    torch.testing.assert_close(torch.tensor(tempered),(logits/.8).log_softmax(-1).gather(1,targets)[:,0])
