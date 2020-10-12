from collections import OrderedDict
import torch.nn as nn

_losses_dict = OrderedDict({
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'categorical_crossentropy': nn.CrossEntropyLoss,
    'ce_loss': nn.CrossEntropyLoss,
    'binary_crossentropy': nn.BCEWithLogitsLoss,
    'bce_loss': nn.BCEWithLogitsLoss,
})

def create_loss_by_name(name):
    name = name.lower()
    if name not in _losses_dict:
        raise KeyError(f'Invalid loss name, we support {list(_losses_dict.keys())}.')
    return _losses_dict[name]()