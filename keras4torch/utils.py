import torch
import pandas as pd
import numpy as np

def to_tensor(*args):
    rt = []
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            arg = torch.from_numpy(arg.values)
        elif isinstance(arg, np.ndarray):
            arg = torch.from_numpy(arg)
        elif isinstance(arg, list):
            arg = torch.tensor(arg)
        elif not isinstance(arg, torch.Tensor):
            raise TypeError('Only DataFrame, ndarray and torch.Tensor are supported.')

        if arg.dtype == torch.float64:
            arg = arg.float()
        elif arg.dtype == torch.int32:
            arg = arg.long()

        rt.append(arg)
                
    return rt[0] if len(rt) == 1 else tuple(rt)