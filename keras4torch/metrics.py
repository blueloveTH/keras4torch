from operator import rshift
import torch
import torch.nn as nn
import math
# For categorical classification
def accuracy(y_pred, y_true):
    right_cnt = (y_pred.argmax(-1) == y_true).sum()
    return right_cnt * 1.0 / y_true.shape[0]

# For binary classification
def binary_accuracy(y_pred, y_true):
    right_cnt = (torch.round(y_pred) == y_true).sum()
    return right_cnt * 1.0 / y_true.shape[0]

def roc_auc(y_pred, y_true):
    pass

def f1_score(y_pred, y_true):
    pass



# For Regression

def mean_squared_error(y_pred, y_true):
    with torch.no_grad():
        y_pred, y_true=torch.tensor(y_pred), torch.tensor(y_true)
        loss = nn.MSELoss()
        return loss(y_pred, y_true).item()


def mean_absolute_error(y_pred, y_true):
    with torch.no_grad():
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
        loss = nn.L1Loss()
        return loss(y_pred, y_true).item()

def mean_absolute_percentage_error(y_pred, y_true):
    with torch.no_grad():
        y_pred, y_true = torch.tensor(y_pred), torch.tensor(y_true)
        return (torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100).item()


def root_mean_squared_error(y_pred, y_true):
    return math.sqrt(mean_squared_error(y_pred, y_true))