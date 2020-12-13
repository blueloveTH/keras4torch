import torch
import torch.nn as nn
import keras4torch as k4t

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout):
        super(_ResidualBlock, self).__init__()
        self.sequential = nn.Sequential(
            self.bn_relu_conv(out_channels, stride=stride, dropout=dropout),
            self.bn_relu_conv(out_channels, stride=1, dropout=dropout)
        )

        self.equalInOut = (in_channels == out_channels)

        if not self.equalInOut:
            self.conv_shortcut = k4t.layers.Conv2d(out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    @staticmethod
    def bn_relu_conv(channels, stride, dropout):
        layers = [k4t.layers.BatchNorm2d(), nn.ReLU(inplace=True)]
        if dropout > 1e-4:
            layers.append(nn.Dropout(dropout, inplace=True))
        layers.append(k4t.layers.Conv2d(channels, kernel_size=3, stride=stride, padding=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.equalInOut:
            return self.conv_shortcut(x) + self.sequential(x)
        else:
            return x + self.sequential(x)

class ResidualBlock(k4t.layers.KerasLayer):
    def build(self, in_shape: torch.Size):
        return _ResidualBlock(in_shape[1], *self.args, **self.kwargs)


def stack_blocks(n, channels, stride, dropout):
    return nn.Sequential(
            *[ResidualBlock(channels, stride if i == 0 else 1, dropout) for i in range(n)]
        )


def wideresnet_cifar10(input_shape, depth, num_classes, widen_factor=10, dropout=0.0):
    """
    WideResNet for CIFAR10/100 implemented in PyTorch.
    This implementation requires less GPU memory than what is required by the official Torch implementation.
    
    It is also applicable to 1x32x32 melspectogram.

    See reference:

    + `Zagoruyko, Sergey, and Nikos Komodakis. "Wide residual networks." arXiv preprint arXiv:1605.07146 (2016).`

    + https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py

    Args:

    * `input_shape` (list or tuple)

    * `depth` (int): A number satisfying `(depth - 4) % 6 == 0`

    * `num_classes` (int): Units of the last fully-connected layer

    * `widen_factor` (int, default=10)

    * `dropout` (float, default=0.0)

    Return: `keras4torch.Model`

    Recommended configs:

    Using `SGD` optimizer with `momentum=0.9` and a `MultistepLR` scheduler.

    Contributor: blueloveTH
    """
    nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
    assert((depth - 4) % 6 == 0)
    n = (depth - 4) // 6

    model = nn.Sequential(
            k4t.layers.Conv2d(nChannels[0], kernel_size=3, stride=1, padding=1, bias=False),
            stack_blocks(n, nChannels[1], stride=1, dropout=dropout),
            stack_blocks(n, nChannels[2], stride=2, dropout=dropout),
            stack_blocks(n, nChannels[3], stride=2, dropout=dropout),

            k4t.layers.BatchNorm2d(), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            k4t.layers.Linear(num_classes)
        )

    model = k4t.Model(model).build(input_shape)

    return model


__all__ = ['wideresnet_cifar10']