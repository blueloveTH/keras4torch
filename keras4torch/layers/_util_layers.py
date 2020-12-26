import torch
import torch.nn as nn
import torch.nn.functional as F

class Lambda(nn.Module):
    """Wrap a function as `nn.Module`."""
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.lambda_func = fn

    def forward(self, *args):
        return self.lambda_func(*args)
        
'''
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
'''

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x):
        return sum(x)

class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)

class Reshape(nn.Module):
    """
    No need to include `batch_size` dimension.
    """
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = list(shape)

    def forward(self, x):
        return x.reshape([x.shape[0]] + self.shape)