from collections import OrderedDict
import sklearn
import torch
import torch.nn.functional as F
from abc import abstractclassmethod

class Metric():
    def __init__(self) -> None:
        pass

    def set_device(self, device):
        self.device = device

    @abstractclassmethod
    def get_abbr(self) -> str:
        pass

class Accuracy(Metric):
    def __init__(self) -> None:
        super(Accuracy, self).__init__()

    def __call__(self, y_pred, y_true):
        if y_pred.shape[-1] == 1:
            return binary_accuracy(y_pred, y_true)
        else:
            return categorical_accuracy(y_pred, y_true)

    def get_abbr(self) -> str:
        return 'acc'

def categorical_accuracy(y_pred, y_true):
    return (y_pred.argmax(-1) == y_true).float().mean()

def binary_accuracy(y_pred, y_true, activation=torch.sigmoid):
    return (torch.round(activation(y_pred)) == y_true).float().mean()


class SklearnMetric(Metric):
    def __init__(self, activation=None) -> None:
        super(SklearnMetric, self).__init__()
        self.activation = activation

        try:
            import sklearn.metrics
        except ImportError:
            raise RuntimeError("This metric requires sklearn to be installed.")

        self.score_fn = self.get_score_fn()

    def __call__(self, y_pred, y_true):
        if self.activation != None:
            y_pred = self.activation(y_pred)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return torch.tensor(self.score_fn(y_true, y_pred), dtype=torch.float32, device=self.device)

    @abstractclassmethod
    def get_score_fn(self):
        pass


class ROC_AUC(SklearnMetric):
    def __init__(self, activation=torch.sigmoid) -> None:
        super(ROC_AUC, self).__init__(activation=activation)

    def get_score_fn(self):
        return sklearn.metrics.roc_auc_score

    def get_abbr(self) -> str:
        return 'auc'

class F1_Score(SklearnMetric):
    def __init__(self, activation=torch.sigmoid) -> None:
        super(F1_Score, self).__init__(activation=lambda x: torch.round(activation(x)))

    def get_score_fn(self):
        return sklearn.metrics.f1_score

    def get_abbr(self) -> str:
        return 'f1'


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


_metrics_dict = OrderedDict({
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'rmse': RootMeanSquaredError,
    'acc': Accuracy,
    'accuracy': Accuracy,
    'auc': ROC_AUC,
    'roc_auc': ROC_AUC,
    'f1': F1_Score,
    'f1_score': F1_Score,
})

def create_metric_by_name(name):
    name = name.lower()
    if name not in _metrics_dict:
        raise KeyError(f'Invalid metric name, we support {list(_metrics_dict.keys())}.')
    return _metrics_dict[name]()