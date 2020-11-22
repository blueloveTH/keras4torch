import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractclassmethod

from torch.nn.modules.conv import Conv1d

class Lambda(nn.Module):
    """Wrap a function as `nn.Module`."""
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.lambda_func = fn

    def forward(self, *args):
        return self.lambda_func(*args)


class SamePadding(nn.Module):
    """
    Pad the output of a module to have the same shape as its input.

    Args:

    * `module` (nn.Module): Target module, usually a convlutional one.

    * `n_dims` (int): Number of dimensions to be pad, if `None` will be infered automatically.

    Usage:
    >>> conv2d = SamePadding(
    ...     nn.Conv2d(16, 32, 3)
    ... )
    >>> input = torch.randn(1, 16, 7, 5)
    >>> output = conv2d(input)
    >>> # The output shape will be equal to the input shape along last `n_dims`
    >>> print(input.shape, output.shape)

    Note:

    If the convlution module has a `stride` > 1, the result may not be what you expect.
    """

    dims_dict = {nn.Conv1d:1, nn.Conv2d:2, nn.Conv3d:3, nn.Linear:1}

    def __init__(self, module, n_dims=None):
        super(SamePadding, self).__init__()
        self.module = module
        self.n_dims = n_dims if n_dims!=None else self.dims_dict[type(module)]
        self.pad_list = None

    def build_pad_list(self, diff_shape):
        self.pad_list = []
        for i in range(self.n_dims):
            d = diff_shape[i]
            if d % 2 == 0:
                self.pad_list += [d//2, d//2]
            else:
                self.pad_list += [d//2, d//2+1]
        self.pad_list = tuple(self.pad_list)

    def forward(self, x):

        def fwd_build(x):
            i_shape = torch.tensor(x.shape)
            x = self.module(x)
            o_shape = torch.tensor(x.shape)
            diff_shape = list(reversed(i_shape - o_shape))
            self.build_pad_list(diff_shape)
            x = F.pad(x, self.pad_list)
            return x

        if self.pad_list == None:
            return fwd_build(x)

        x = self.module(x)
        x = F.pad(x, self.pad_list)
        return x

class KerasLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(KerasLayer, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.module = None

    @abstractclassmethod
    def build(self, in_shape: torch.Size):
        pass

    def forward(self, x):
        if not self.module:
            self.module = self.build(x.shape)
        return self.module.forward(x)

# Conv
class Conv1d(KerasLayer):
    def build(self, in_shape):
        return nn.Conv1d(in_shape[1], *self.args, **self.kwargs)

class Conv2d(KerasLayer):
    def build(self, in_shape):
        return nn.Conv2d(in_shape[1], *self.args, **self.kwargs)

class Conv3d(KerasLayer):
    def build(self, in_shape):
        return nn.Conv3d(in_shape[1], *self.args, **self.kwargs)

# BatchNorm
class BatchNorm1d(KerasLayer):
    def build(self, in_shape):
        return nn.BatchNorm1d(in_shape[1], *self.args, **self.kwargs)

class BatchNorm2d(KerasLayer):
    def build(self, in_shape):
        return nn.BatchNorm2d(in_shape[1], *self.args, **self.kwargs)

class BatchNorm3d(KerasLayer):
    def build(self, in_shape):
        return nn.BatchNorm3d(in_shape[1], *self.args, **self.kwargs)

class Linear(KerasLayer):
    def build(self, in_shape):
        return nn.Linear(in_shape[1], *self.args, **self.kwargs)


class _SqueezeAndExcitation1d(nn.Module):
    def __init__(self, in_shape, reduction_ratio=16, channel_last=False):
        super(_SqueezeAndExcitation1d, self).__init__()

        C_dim = 2 if channel_last else 1
        L_dim = 1 if channel_last else 2
        C_in = in_shape[C_dim]

        self.fc = nn.Sequential(
            Lambda(lambda x: x.mean(dim=L_dim)),
            Linear(C_in // reduction_ratio), nn.ReLU(),
            Linear(C_in), nn.Sigmoid(),
            Lambda(lambda x: x.unsqueeze(L_dim))
        )
    
    def forward(self, x):
        return x * self.fc(x)

class SqueezeAndExcitation1d(KerasLayer):
    """
    Squeeze-and-Excitation Module

    input: [N, C_in, L_in] by default and [N, L_in, C_in] if `channel_last=True`

    output: The same with the input
    """
    def build(self, in_shape):
        return _SqueezeAndExcitation1d(in_shape, *self.args, **self.kwargs)