import torch.nn as nn

from ._keras_layers import Lambda, KerasLayer, Linear

class _SqueezeAndExcitation1d(nn.Module):
    def __init__(self, in_shape, reduction_ratio=16, attn_dim=1):
        super(_SqueezeAndExcitation1d, self).__init__()

        assert attn_dim == 1 or attn_dim == 2
        self.attn_dim = attn_dim

        self.fc = nn.Sequential(
            Lambda(lambda x: x.mean(dim=2-attn_dim)),
            Linear(in_shape[attn_dim] // reduction_ratio), nn.ReLU(),
            Linear(in_shape[attn_dim]), nn.Sigmoid(),
            Lambda(lambda x: x.unsqueeze(2-attn_dim))
        )
    
    def forward(self, x):
        return x * self.fc(x)


class SqueezeAndExcitation1d(KerasLayer):
    """
    Squeeze-and-Excitation Module 1D

    See reference: `Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.`

    Args:

    * `reduction_ratio` (int, default=16)

    * `attn_dim` (bool, default=2)

    Input: [N, C_in, L_in] by default and [N, L_in, C_in] if `attn_dim=2`

    Output: The same with the input

    Contributor: blueloveTH
    """
    def build(self, in_shape):
        return _SqueezeAndExcitation1d(in_shape, *self.args, **self.kwargs)

__all__ = ['SqueezeAndExcitation1d']