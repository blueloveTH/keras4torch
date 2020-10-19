from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractclassmethod

from torch.nn.modules import activation

class Metric():
    def __init__(self) -> None:
        pass
    @abstractclassmethod
    def get_abbr(self) -> str:
        pass

class Accuracy(Metric):
    def __init__(self) -> None:
        super(Accuracy, self).__init__()

    def __call__(self, y_pred, y_true):
        if y_true.shape[-1] == 1:
            return binary_accuracy(y_pred, y_true)
        else:
            return categorical_accuracy(y_pred, y_true)

    def get_abbr(self) -> str:
        return 'acc'

def categorical_accuracy(y_pred, y_true):
    right_cnt = (y_pred.argmax(-1) == y_true).sum()
    return right_cnt.float() / y_true.shape[0]

def binary_accuracy(y_pred, y_true, activation=torch.sigmoid):
    right_cnt = (torch.round(activation(y_pred)) == y_true).sum()
    return right_cnt.float() / y_true.shape[0]


class ROC_AUC(Metric):
    def __init__(self, activation=torch.sigmoid) -> None:
        super(ROC_AUC, self).__init__()
        if activation == None:
            activation = lambda x: x
        self.activation = activation

        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            raise RuntimeError("This metric requires sklearn to be installed.")

        self.score_fn = roc_auc_score

    def __call__(self, y_pred, y_true):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return torch.FloatTensor(self.score_fn(y_true, y_pred))

    def get_abbr(self) -> str:
        return 'auc'


# For Regression

class MeanSquaredError(Metric):
    def __init__(self) -> None:
        super(MeanSquaredError, self).__init__()

    def __call__(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def get_abbr(self) -> str:
        return 'mse'

class MeanAbsoluteError(Metric):
    def __init__(self) -> None:
        super(MeanAbsoluteError, self).__init__()

    def __call__(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)

    def get_abbr(self) -> str:
        return 'mae'

class RootMeanSquaredError(Metric):
    def __init__(self) -> None:
        super(RootMeanSquaredError, self).__init__()

    def __call__(self, y_pred, y_true):
        return torch.sqrt(F.mse_loss(y_pred, y_true))

    def get_abbr(self) -> str:
        return 'rmse'


'''
def mean_absolute_percentage_error(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
'''


_metrics_dict = OrderedDict({
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'rmse': RootMeanSquaredError,
    'acc': Accuracy,
    'accuracy': Accuracy,
    'auc': ROC_AUC,
    'roc_auc': ROC_AUC
})

def create_metric_by_name(name):
    name = name.lower()
    if name not in _metrics_dict:
        raise KeyError(f'Invalid metric name, we support {list(_metrics_dict.keys())}.')
    return _metrics_dict[name]()