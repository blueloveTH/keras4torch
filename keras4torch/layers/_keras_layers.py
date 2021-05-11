import torch
import torch.nn as nn
from abc import abstractclassmethod

from ._util_layers import Lambda


class KerasLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(KerasLayer, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.module = None

    @abstractclassmethod
    def build(self, in_shape: torch.Size):
        pass

    def forward(self, x, *args, **kwargs):
        if self.module is None:
            self.module = self.build(x.shape)
            self.module._k4t_layer_tag = 0
        return self.module(x, *args, **kwargs)

    @property
    def is_built(self):
        return self.module is not None

    def apply(self, fn):
        if self.module is None:
            raise AssertionError("This module hasn't been built yet.")
        return self.module.apply(fn)



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
        return nn.Linear(in_shape[-1], *self.args, **self.kwargs)

class LayerNorm(KerasLayer):
    def build(self, in_shape):
        return nn.LayerNorm(in_shape[-1], *self.args, **self.kwargs)


class GRU(KerasLayer):
    def __init__(self, *args, return_sequences=True, batch_first=True, **kwargs):
        super(GRU, self).__init__(*args, **kwargs)
        self.batch_first = batch_first
        self.return_sequences = return_sequences

    # [batch, seq, feature] as input
    def build(self, in_shape):
        layers = [nn.GRU(in_shape[-1], *self.args, batch_first=self.batch_first, **self.kwargs)]
        if self.return_sequences:
            layers.append(Lambda(lambda x: x[0]))
        return nn.Sequential(*layers)

class LSTM(KerasLayer):
    def __init__(self, *args, return_sequences=True, batch_first=True, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)
        self.batch_first = batch_first
        self.return_sequences = return_sequences

    # [batch, seq, feature] as input
    def build(self, in_shape):
        layers = [nn.LSTM(in_shape[-1], *self.args, batch_first=self.batch_first, **self.kwargs)]
        if self.return_sequences:
            layers.append(Lambda(lambda x: x[0]))
        return nn.Sequential(*layers)
