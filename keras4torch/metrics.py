import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractclassmethod

class Metric():
    def __init__(self) -> None:
        pass
    '''
    @abstractclassmethod
    def get_abbr(self) -> str:
        pass
    '''

# For categorical classification
class Accuracy(Metric):
    def __init__(self) -> None:
        super(Accuracy, self).__init__()

    def __call__(self, y_pred, y_true):
        right_cnt = (y_pred.argmax(-1) == y_true).sum()
        return right_cnt * 1.0 / y_true.shape[0]

class BinaryAccuracy(Metric):
    def __init__(self) -> None:
        super(BinaryAccuracy, self).__init__()

    def __call__(self, y_pred, y_true):
        right_cnt = (torch.round(y_pred) == y_true).sum()
        return right_cnt * 1.0 / y_true.shape[0]

class ROC_AUC(Metric):
    def __init__(self) -> None:
        super(ROC_AUC, self).__init__()
        raise NotImplementedError()

    def __call__(self, y_pred, y_true):
        pass



# For Regression

class MeanSquaredError(Metric):
    def __init__(self) -> None:
        super(MeanSquaredError, self).__init__()

    def __call__(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

class MeanAbsoluteError(Metric):
    def __init__(self) -> None:
        super(MeanAbsoluteError, self).__init__()

    def __call__(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)

class RootMeanSquaredError(Metric):
    def __init__(self) -> None:
        super(RootMeanSquaredError, self).__init__()

    def __call__(self, y_pred, y_true):
        return torch.sqrt(F.mse_loss(y_pred, y_true))


'''
def mean_absolute_percentage_error(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
'''