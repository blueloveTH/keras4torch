from collections import OrderedDict
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


class Mish(nn.Module):
    """
    `mish(x) = x * tanh(softplus(x))`

    See reference:
    
    + `Misra, Diganta. "Mish: A self regularized non-monotonic neural activation function." arXiv preprint arXiv:1908.08681 (2019).`

    + https://github.com/digantamisra98/Mish
    """
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Swish(nn.Module):
    """
    `swish(x) = x * sigmoid(beta * x)` where `beta` is a trainable parameter.

    See reference:
    + `Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Swish: a self-gated activation function." arXiv preprint arXiv:1710.05941 7 (2017).`

    + https://en.wikipedia.org/wiki/Swish_function
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)



_activations_dict = OrderedDict({
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(dim=-1),
    'selu': nn.SELU(),
    'celu': nn.CELU(),
    'leaky_relu': nn.LeakyReLU(),
    'relu6': nn.ReLU6(),
    'elu': nn.ELU(),
    'sigmoid': nn.Sigmoid(),
    'mish': Mish(),
    'swish': Swish(),
})


def _create_activation(i):
    if isinstance(i, str):
        name = i.lower()
        if name not in _activations_dict:
            raise KeyError(f'Invalid name, we support {list(_activations_dict.keys())}.')
        return deepcopy(_activations_dict[name])
    else:
        return i

__all__ = ['Mish', 'Swish']