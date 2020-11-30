import torch
from collections import OrderedDict

_optimizers_dict = OrderedDict({
    'adam': (torch.optim.Adam, 1e-3),
    'rmsprop': (torch.optim.RMSprop, 1e-3),
    'sgd': (torch.optim.SGD, 1e-2)
})

def _create_optimizer(i, model_parameters):
    if isinstance(i, str):
        name = i.lower()
        if name not in _optimizers_dict:
            raise KeyError(f'Invalid name, we support {list(_optimizers_dict.keys())}.')
        optim_class, lr = _optimizers_dict[name]
        return optim_class(model_parameters, lr=lr)
    else:
        return i

__all__ = []