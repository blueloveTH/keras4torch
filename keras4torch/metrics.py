import torch

def accuracy(y_pred, y_true):
    right_cnt = (y_pred.argmax(-1) == y_true).sum()
    return right_cnt * 1.0 / y_true.shape[0]