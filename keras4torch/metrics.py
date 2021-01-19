from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np

class Metric():
    def __init__(self) -> None:
        pass

    def get_abbr(self) -> str:
        raise NotImplementedError()

class Accuracy(Metric):
    def __call__(self, y_pred, y_true):
        if y_pred.shape[-1] == 1 or y_pred.dim() == 1:
            return binary_accuracy(y_pred, y_true)
        return categorical_accuracy(y_pred, y_true)

    def get_abbr(self) -> str:
        return 'acc'

def categorical_accuracy(y_pred, y_true):
    return (y_pred.argmax(-1) == y_true).float().mean()

def binary_accuracy(y_pred, y_true, activation=torch.sigmoid):
    return (torch.round(activation(y_pred)) == y_true).float().mean()



class ROC_AUC(Metric):
    """
    Fast AUC implementation collected from Kaggle.
    
    By default, this metric will output the same result as `sklearn.metrics.roc_auc_score`

    See reference: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    """
    def __init__(self, activation=torch.sigmoid):
        super(ROC_AUC, self).__init__()
        from scipy.stats import rankdata
        self.sort_fn = rankdata
        self.activation = activation

    @staticmethod
    def _auc(y_true, pred_ranks):
        n_pos = np.sum(y_true).astype('int64')
        n_neg = len(y_true) - n_pos
        left = pred_ranks[y_true==1].sum() - n_pos * (n_pos + 1) / 2
        return left / (n_pos * n_neg)

    def __call__(self, y_pred, y_true):
        y_true = y_true.cpu().numpy().reshape(-1)
        y_pred = self.activation(y_pred).cpu().numpy().reshape(-1)
        pred_ranks = self.sort_fn(y_pred)
        result = self._auc(y_true, pred_ranks)
        return torch.tensor(result, dtype=torch.float32, device='cpu')

    def get_abbr(self) -> str:
        return 'auc'



class SklearnMetric(Metric):
    def __init__(self, activation=None) -> None:
        super(SklearnMetric, self).__init__()
        self.activation = activation

        try:
            from sklearn import metrics as sklearn_metrics
        except ImportError:
            raise RuntimeError("This metric requires sklearn to be installed.")

        self.score_fn = self.get_score_fn(sklearn_metrics)

    def __call__(self, y_pred, y_true):
        if self.activation is not None:
            y_pred = self.activation(y_pred)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return torch.tensor(self.score_fn(y_true, y_pred), dtype=torch.float32, device='cpu')

    def get_score_fn(self, sklearn_metrics):
        raise NotImplementedError()


class ROC_AUC_2(SklearnMetric):
    def __init__(self, activation=torch.sigmoid) -> None:
        super(ROC_AUC_2, self).__init__(activation=activation)

    def get_score_fn(self, sklearn_metrics):
        return sklearn_metrics.roc_auc_score

    def get_abbr(self) -> str:
        return 'auc'


class F1_Score(SklearnMetric):
    def __init__(self, activation=torch.sigmoid) -> None:
        super(F1_Score, self).__init__(activation=lambda x: torch.round(activation(x)))

    def get_score_fn(self, sklearn_metrics):
        return sklearn_metrics.f1_score

    def get_abbr(self) -> str:
        return 'f1'


# For Regression

class MeanSquaredError(Metric):
    def __call__(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def get_abbr(self) -> str:
        return 'mse'

class MeanAbsoluteError(Metric):
    def __call__(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)

    def get_abbr(self) -> str:
        return 'mae'

class RootMeanSquaredError(Metric):
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

def _create_metric(i):
    if isinstance(i, str):
        name = i.lower()
        if name not in _metrics_dict:
            raise KeyError(f'Invalid name, we support {list(_metrics_dict.keys())}.')
        return _metrics_dict[name]()
    return i


def _to_metrics_dic(metrics):
    m = OrderedDict()
    if isinstance(metrics, dict):
        m.update(metrics)
    elif isinstance(metrics, list):
        for tmp_m in metrics:
            tmp_m = _create_metric(tmp_m)
            if isinstance(tmp_m, Metric):
                m[tmp_m.get_abbr()] = tmp_m
            elif callable(tmp_m):
                m[tmp_m.__name__] = tmp_m
            else:
                raise TypeError('Unsupported type.')
    elif not (metrics is None):
        raise TypeError('Argument `metrics` should be either a dict or list.')
    return m

__all__ = ['Metric', 'Accuracy', 'categorical_accuracy', 'binary_accuracy',
                'SklearnMetric', 'ROC_AUC', 'F1_Score', 'MeanSquaredError', 'MeanAbsoluteError', 'RootMeanSquaredError']