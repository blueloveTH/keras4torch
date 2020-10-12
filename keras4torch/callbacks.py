import abc
from abc import abstractclassmethod
from enum import Enum

import torch

class Events(Enum):
    ON_EPOCH_END = 'on_epoch_end'
    ON_EPOCH_BEGIN = 'on_epoch_begin'
    ON_TRAIN_BEGIN = 'on_train_begin'
    ON_TRAIN_END = 'on_train_end'

class Callback():
    def __init__(self) -> None:
        pass
    
    @abstractclassmethod
    def get_callbacks_dict(self):
        pass


class ModelCheckpoint(Callback):
    def __init__(self) -> None:
        super(ModelCheckpoint, self).__init__()

        raise NotImplementedError()

    def get_callbacks_dict(self):
        return {Events.ON_EPOCH_END: self.on_epoch_end}

    def on_epoch_end(self, trainer):
        pass


class EarlyStopping(Callback):
    def __init__(self) -> None:
        super(EarlyStopping, self).__init__()

        raise NotImplementedError()

    def get_callbacks_dict(self):
        return {Events.ON_EPOCH_END: self.on_epoch_end}

    def on_epoch_end(self, trainer):
        pass



class LRScheduler(Callback):
    def __init__(self, lr_scheduler) -> None:
        super(LRScheduler, self).__init__()
        self.lr_scheduler = lr_scheduler

    def get_callbacks_dict(self):
        return {Events.ON_EPOCH_END: self.on_epoch_end}

    def on_epoch_end(self, trainer):
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


