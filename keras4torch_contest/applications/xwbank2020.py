import torch
import torch.nn as nn
import keras4torch as k4t

class res_block(nn.Module):
    def __init__(self, filters, kernel_size):
        super(res_block, self).__init__()

        self.conv1d = nn.Sequential(
            k4t.layers.Conv1d(filters, 1),
            nn.ReLU(inplace=True), k4t.layers.BatchNorm1d(),

            k4t.layers.Conv1d(filters, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True), k4t.layers.BatchNorm1d(),

            k4t.layers.Conv1d(filters, 1),
            nn.ReLU(inplace=True), k4t.layers.BatchNorm1d(),
        )

        self.shortcut = k4t.layers.Conv1d(filters, 1)
    
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


def conv1d_xwbank2020(input_shape, num_classes, compile=False):
    """
    Conv1D model for time series classification
    """
    model = k4t.Model(_Conv1D_xwbank2020(num_classes=num_classes))
    model.build(input_shape)

    def weights_init(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    model.apply(weights_init)

    if compile:
        model.compile(optimizer='adam', loss=k4t.losses.CELoss(label_smoothing=0.1), metrics=['acc'])

    return model

class _Conv1D_xwbank2020(nn.Module):
    def __init__(self, num_classes):
        super(_Conv1D_xwbank2020, self).__init__()

        self.seq_3 = inception_block(kernel_size=3)
        self.seq_5 = inception_block(kernel_size=5)
        self.seq_7 = inception_block(kernel_size=7)
        
        self.dense = nn.Sequential(
            k4t.layers.Linear(512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            k4t.layers.Linear(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            k4t.layers.Linear(num_classes)
        )

    def forward(self, x):
        x = torch.cat([self.seq_3(x), self.seq_5(x), self.seq_7(x)], axis=-1)
        return self.dense(x)