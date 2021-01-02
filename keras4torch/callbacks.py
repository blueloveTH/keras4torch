import numpy as np
from ._training import Trainer, StopTrainingError, Events
from .metrics import _to_metrics_dic

class Callback():
    def __init__(self) -> None:
        pass
    
    def get_callbacks_dict(self):
        return {i: getattr(self, i.value) for i in Events if hasattr(self, i.value)}
        

def _guess_auto_mode(monitor, mode) -> str:
    if mode != 'auto':
        assert mode == 'max' or mode == 'min'
        return mode
    for s in ['loss', 'mse', 'mae', 'mape', 'error', 'err']:
        if s in monitor:
            return 'min'
    return 'max'

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='auto', save_best_only=True, save_weights_only=True, verbose=0):
        super(ModelCheckpoint, self).__init__()

        if not (save_weights_only):
            raise ValueError('`ModelCheckpoint` only supports `save_weights_only=True`.')

        self.filepath = filepath
        self.monitor = monitor
        self.mode = _guess_auto_mode(monitor, mode)
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best_score = -np.inf if self.mode == 'max' else np.inf
        
    def is_better(self, curr_score):
        if self.mode == 'max':
            return curr_score > self.best_score
        else:   # mode == 'min'
            return curr_score < self.best_score

    def on_epoch_end(self, trainer: Trainer):
        filename = self.filepath.format(epoch=trainer.epoch, **trainer.logger.metrics)

        if not self.save_best_only:
            trainer.model.save_weights(filename)
            return

        if self.monitor not in trainer.logger.metrics:
            raise KeyError(f'No such metric: {self.monitor}.')

        curr_score = trainer.logger.metrics[self.monitor]
        if self.is_better(curr_score):
            self.best_score = curr_score
            if self.verbose == 1:
                print("[INFO] Model saved at '{}'. The best score is {:.4f}.".format(self.filepath, self.best_score))
            trainer.model.save_weights(filename)


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', mode='auto', min_delta=0, patience=5, baseline=None, verbose=1) -> None:
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.mode = _guess_auto_mode(monitor, mode)
        self.min_delta = min_delta
        self.chances = self.patience = patience
        self.baseline = baseline
        self.verbose = verbose
        self.baseline_flag = False

        self.best_score = -np.inf if self.mode == 'max' else np.inf

    def is_better(self, curr_score):
        if self.mode == 'max':
            return curr_score - self.best_score > self.min_delta
        else:   # mode == 'min'
            if self.baseline != None and curr_score > self.baseline:
                return False
            return self.best_score - curr_score > self.min_delta

    def is_surpass_baseline(self, curr_score):
        if self.baseline == None or self.baseline_flag:
            return True
        if self.mode == 'max' and curr_score > self.baseline:
            self.baseline_flag = True
            return True
        if self.mode == 'min' and curr_score < self.baseline:
            self.baseline_flag = True
            return True
        return False

    def on_epoch_end(self, trainer: Trainer):
        if self.monitor not in trainer.logger.metrics:
            raise KeyError(f'No such metric: {self.monitor}.')

        curr_score = trainer.logger.metrics[self.monitor]

        if not self.is_surpass_baseline(curr_score):
            return

        if self.is_better(curr_score):
            self.best_score = curr_score
            self.chances = self.patience
        else:
            self.chances -= 1
            if self.chances <= 0:
                if self.verbose == 1:
                    print('[INFO] Early Stopped. The best score is {:.4f}.'.format(self.best_score))
                raise StopTrainingError




class LRScheduler(Callback):
    def __init__(self, lr_scheduler) -> None:
        super(LRScheduler, self).__init__()
        self.lr_scheduler = lr_scheduler

    def on_epoch_end(self, trainer: Trainer):
        self.lr_scheduler.step()



class LambdaCallback(Callback):
    def __init__(self, on_epoch_begin=None, on_epoch_end=None, on_train_begin=None, on_train_end=None) -> None:
        super(LambdaCallback, self).__init__()

        self.callbacks_dict = {}

        if on_epoch_begin:
            self.callbacks_dict.update({Events.ON_EPOCH_BEGIN: on_epoch_begin})
        if on_epoch_end:
            self.callbacks_dict.update({Events.ON_EPOCH_END: on_epoch_end})
        if on_train_begin:
            self.callbacks_dict.update({Events.ON_TRAIN_BEGIN: on_train_begin})
        if on_train_end:
            self.callbacks_dict.update({Events.ON_TRAIN_END: on_train_end})

    def get_callbacks_dict(self):
        return self.callbacks_dict


__all__ = ['Callback', 'ModelCheckpoint', 'EarlyStopping', 'LRScheduler', 'LambdaCallback']