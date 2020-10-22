from collections import OrderedDict
import torch
import time
import numpy as np
import pandas as pd
from enum import Enum

from torch._C import device

class Events(Enum):
    ON_EPOCH_END = 'on_epoch_end'
    ON_EPOCH_BEGIN = 'on_epoch_begin'
    ON_TRAIN_BEGIN = 'on_train_begin'
    ON_TRAIN_END = 'on_train_end'


class StopTrainingError(Exception):
    def __init__(self):
        pass

class Trainer(object):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = Logger(self)
    
    def register_callbacks(self, callbacks):
        self.event_dict = {Events(k).value: list() for k in Events}

        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for c in callbacks:
            for key, func in c.get_callbacks_dict().items():
                self.event_dict[Events(key).value] += [func]
    
    def __fire_event(self, key):
        func_list = self.event_dict[Events(key).value]
        for func in func_list:
            func(self)

    def run(self, train_loader, val_loader, max_epochs, verbose, precise_mode):
        self.max_epochs = max_epochs
        self.logger.verbose = verbose
        self.logger.on_train_begin(train_loader, val_loader)
        self.__fire_event(Events.ON_TRAIN_BEGIN)

        for epoch in range(1, max_epochs+1):
            self.epoch = epoch
            self.logger.on_epoch_begin()
            self.__fire_event(Events.ON_EPOCH_BEGIN)

            if precise_mode:
                self.train_precise_mode(train_loader)
                train_metrics = self.evaluate(train_loader) 
            else:
                train_metrics = self.train_fast_mode(train_loader)

            val_metrics = self.evaluate(val_loader) if val_loader else {}

            self.logger.on_epoch_end(epoch=epoch, max_epochs=max_epochs, train_metrics=train_metrics, val_metrics=val_metrics)

            try:
                self.__fire_event(Events.ON_EPOCH_END)
            except StopTrainingError:
                break

        self.__fire_event(Events.ON_TRAIN_END)
        return self.logger.history

    def train_precise_mode(self, data_loader):
        self.model.train()
        for x_batch, y_batch in data_loader:
            x = x_batch.to(device=self.device)
            y = y_batch.to(device=self.device)
            self.optimizer.zero_grad()
            y_pred = self.model.forward(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()

    def train_fast_mode(self, data_loader):
        self.model.train()
        score_funcs = list(self.metrics.values())[1:]
        metrics = []
        for x_batch, y_batch in data_loader:
            x = x_batch.to(device=self.device)
            y = y_batch.to(device=self.device)
            self.optimizer.zero_grad()
            y_pred = self.model.forward(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                batch_metrics = [loss]
                for score_fn in score_funcs:
                    batch_metrics.append(score_fn(y_pred, y))
                metrics.append(torch.tensor(batch_metrics))

        metrics = torch.stack(metrics).mean(dim=0).cpu().numpy()
        return OrderedDict({k:v for k,v in zip(self.metrics.keys(), metrics)})

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()

        metrics = []
        y_true = []
        y_pred = []

        for x_batch, y_batch in data_loader:
            x = x_batch.to(device=self.device)
            y = y_batch.to(device=self.device)
            y_true.append(y)
            y_pred.append(self.model.forward(x))

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        for score_fn in self.metrics.values():
            metrics.append(score_fn(y_pred, y_true).mean(dim=0).cpu().numpy())
        
        return OrderedDict({k:v for k,v in zip(self.metrics.keys(), metrics)})



class Logger(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.verbose = 0

    def on_train_begin(self, train_loader, val_loader):
        if val_loader != None:
            print(f'Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples:')
        else:
            print(f'Train on {len(train_loader.dataset)} samples:')

    def on_epoch_begin(self, **kwargs):
        self.time = time.time()

    def on_epoch_end(self, **kwargs):
        if not hasattr(self, 'history'):
            train_columns = [key for key in kwargs['train_metrics'].keys()]
            val_columns = ['val_'+key for key in kwargs['val_metrics'].keys()]
            self.history = pd.DataFrame(columns=train_columns + val_columns)
        
        time_elapsed = time.time() - self.time

        content = []
        content.append(f"Epoch {kwargs['epoch']}/{kwargs['max_epochs']}")

        time_str = f'{round(time_elapsed, 1)}s' if time_elapsed < 10 else f'{int(time_elapsed + 0.5)}s'
        content.append(time_str)
        
        self.metrics = OrderedDict()
 
        for k, v in kwargs['train_metrics'].items():
            content.append(f'{k}: ' + '{:.4f}'.format(v)) 
            self.metrics[k] = v

        for k, v in kwargs['val_metrics'].items():
            k = 'val_' + k
            content.append(f'{k}: ' + '{:.4f}'.format(v))
            self.metrics[k] = v

        self.metrics['lr'] = self.trainer.optimizer.param_groups[0]['lr']
        content.append('lr: {:.0e}'.format(self.metrics['lr']))

        self.history.loc[kwargs['epoch']] = self.metrics

        if self.verbose == 1:
            print(' - '.join(content))
        elif self.verbose == 2:
            print('|'.join(content).replace(' ', '').replace('Epoch', ''))