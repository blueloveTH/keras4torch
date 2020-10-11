from operator import rshift
import torch

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
    pass

def mean_absolute_error(y_pred, y_true):
    pass

def mean_absolute_percentage_error(y_pred, y_true):
    pass

def root_mean_square_error(y_pred, y_true):
    pass