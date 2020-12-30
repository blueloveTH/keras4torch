import torch
import numpy as np
from multiprocessing import cpu_count

def _to_tensor_leaf(arg):
    if isinstance(arg, np.ndarray):
        arg = torch.from_numpy(arg)
    elif not isinstance(arg, torch.Tensor):
        raise TypeError(f'Only np.ndarray and torch.Tensor are supported. Got {type(arg)}.')

    if arg.dtype == torch.float64:
        print('\033[33m' + '[Warning] Auto convert float64 to float32, this could lead to extra memory usage.')
        arg = arg.float()
    elif arg.dtype == torch.int32:
        print('\033[33m' + '[Warning] Auto convert int32 to int64, this could lead to extra memory usage.')
        arg = arg.long()
    return arg

def _deep_to_tensor(arg):
    if isinstance(arg, list) or isinstance(arg, tuple):
        return type(arg)([_deep_to_tensor(a) for a in arg])
    else:
        return _to_tensor_leaf(arg)

def to_tensor(*args):
    rt = [_deep_to_tensor(arg) for arg in args]            
    return rt[0] if len(rt) == 1 else tuple(rt)


def _get_num_workers(num_workers):
    if num_workers == -1:
        return cpu_count() - 1
    return num_workers