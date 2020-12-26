import torch.nn as nn

from ._keras_layers import *

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
    Squeeze-and-Excitation Module 1D

    See reference: `Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.`

    Args:

    * `reduction_ratio` (int, default=16)

    * `channel_last` (bool, default=False)

    Input: [N, C_in, L_in] by default and [N, L_in, C_in] if `channel_last=True`

    Output: The same with the input

    Contributor: blueloveTH
    """
    def build(self, in_shape):
        return _SqueezeAndExcitation1d(in_shape, *self.args, **self.kwargs)

__all__ = ['SqueezeAndExcitation1d']