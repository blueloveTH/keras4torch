from collections import OrderedDict
import torch.nn as nn
from copy import deepcopy

_activations_dict = OrderedDict({
    'relu': nn.ReLU(inplace=True),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(dim=-1),
    'selu': nn.SELU(inplace=True),
    'celu': nn.CELU(inplace=True),
    'leaky_relu': nn.LeakyReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'elu': nn.ELU(inplace=True),
    'sigmoid': nn.Sigmoid(),
})

def _create_activation(i):
    if isinstance(i, str):
        name = i.lower()
        if name not in _activations_dict:
            raise KeyError(f'Invalid metric name, we support {list(_activations_dict.keys())}.')
        return deepcopy(_activations_dict[name])
    else:
        return i