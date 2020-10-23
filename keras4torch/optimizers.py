import torch
from collections import OrderedDict

_optimizers_dict = OrderedDict({
    'adam': torch.optim.Adam,
    'rmsprop': torch.optim.RMSprop,
    'sgd': torch.optim.SGD
})

def create_optimizer_by_name(name, *args):
    name = name.lower()
    if name not in _optimizers_dict:
        raise KeyError(f'Invalid metric name, we support {list(_optimizers_dict.keys())}.')
    if name == 'sgd':
        lr = 1e-2
    else:
        lr = 1e-3
    return _optimizers_dict[name](*args, lr)