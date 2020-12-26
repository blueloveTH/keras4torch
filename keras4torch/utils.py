import torch
import numpy as np
from multiprocessing import cpu_count

def to_tensor(*args):
    rt = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = torch.from_numpy(arg)
        elif not isinstance(arg, torch.Tensor):
            raise TypeError('Only np.ndarray and torch.Tensor are supported.')

        if arg.dtype == torch.float64:
            print('[Warning] Auto convert float64 to float32, this could lead to extra memory usage.')
            arg = arg.float()
        elif arg.dtype == torch.int32:
            print('[Warning] Auto convert int32 to int64, this could lead to extra memory usage.')
            arg = arg.long()

        rt.append(arg)
                
    return rt[0] if len(rt) == 1 else tuple(rt)


def _get_num_workers(num_workers):
    if num_workers == -1:
        return cpu_count() - 1
    else:
        return min(num_workers, cpu_count())