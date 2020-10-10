import torch

class Lambda(torch.nn.Module):
    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.lambda_func = fn

    def forward(self, *args):
        return self.lambda_func(*args)