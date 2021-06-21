import torch.nn as nn

class Lambda(nn.Module):
    """Wrap a function as `nn.Module`."""
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.lambda_func = fn

    def forward(self, *args):
        return self.lambda_func(*args)