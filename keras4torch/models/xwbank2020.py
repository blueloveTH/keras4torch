from ..models import Model
import torch
import torch.nn as nn
from .. import layers

class res_block(nn.Module):
    def __init__(self, filters, kernel_size):
        super(res_block, self).__init__()

        self.conv1d = nn.Sequential(
            layers.Conv1d(filters, 1),
            nn.ReLU(), layers.BatchNorm1d(),

            layers.Conv1d(filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(), layers.BatchNorm1d(),

            layers.Conv1d(filters, 1),
            nn.ReLU(), layers.BatchNorm1d(),
        )

        self.shortcut = layers.Conv1d(filters, 1)
    
    def forward(self, x):
        return self.conv1d(x) + self.shortcut(x)


class inception_block(nn.Module):
    def __init__(self, filters=128, kernel_size=5):
        super(inception_block, self).__init__()

        self.block = nn.Sequential(
            res_block(filters, kernel_size),
            nn.MaxPool1d(2, 2),       # pass
            nn.Dropout2d(0.3),        # pass
            res_block(filters//2, kernel_size),
            nn.AdaptiveAvgPool1d(1),  # pass
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


def conv1d_xwbank2020(*args, **kwargs):
    """
    Conv1D model for time series classification

    params: num_classes
    """
    return Model(_Conv1D_xwbank2020(*args, **kwargs))


class _Conv1D_xwbank2020(nn.Module):
    def __init__(self, num_classes):
        super(_Conv1D_xwbank2020, self).__init__()

        self.seq_3 = inception_block(kernel_size=3)
        self.seq_5 = inception_block(kernel_size=5)
        self.seq_7 = inception_block(kernel_size=7)
        
        self.dense = nn.Sequential(
            layers.Linear(512), nn.ReLU(), nn.Dropout(0.3),
            layers.Linear(128), nn.ReLU(), nn.Dropout(0.3),
            layers.Linear(num_classes)
        )

    def forward(self, x):
        x = torch.cat([self.seq_3(x), self.seq_5(x), self.seq_7(x)], axis=-1)
        return self.dense(x)