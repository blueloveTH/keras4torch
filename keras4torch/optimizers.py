import torch
from collections import OrderedDict

_optimizers_dict = OrderedDict({
    'adam': (torch.optim.Adam, {'lr': 1e-3}),
    'rmsprop': (torch.optim.RMSprop, {'lr': 1e-3}),
    'sgd': (torch.optim.SGD, {'lr': 1e-2, 'momentum': 0.9})
})

def _create_optimizer(i, model_parameters):
    if isinstance(i, str):
        name = i.lower()
        if name not in _optimizers_dict:
            raise KeyError(f'Invalid name, we support {list(_optimizers_dict.keys())}.')
        optim_class, kwargs = _optimizers_dict[name]
        return optim_class(model_parameters, **kwargs)
    else:
        return i

__all__ = []