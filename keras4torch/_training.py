import torch
import time
import pandas as pd
from enum import Enum

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

    def run(self, train_loader, val_loader, max_epochs, verbose, use_amp):
        self.use_amp = use_amp if self.device == 'cuda' else False
        self.max_epochs = max_epochs
        self.logger.on_train_begin(verbose, train_loader, val_loader)
        self.__fire_event(Events.ON_TRAIN_BEGIN)

        for epoch in range(1, max_epochs+1):
            self.epoch = epoch
            self.logger.on_epoch_begin()
            self.__fire_event(Events.ON_EPOCH_BEGIN)

            train_metrics = self.train_on_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, self.use_amp) if val_loader else {}

            self.logger.on_epoch_end(epoch=epoch, max_epochs=max_epochs, train_metrics=train_metrics, val_metrics=val_metrics)

            try:
                self.__fire_event(Events.ON_EPOCH_END)
            except StopTrainingError:
                break

        self.__fire_event(Events.ON_TRAIN_END)
        return self.logger.history

    @torch.no_grad()
    def __calc_metrics(self, y_pred, y_true):
        metrics = {}
        for key, score_fn in self.metrics.items():
            metrics[key] = score_fn(y_pred, y_true).mean().cpu().item()
        return metrics


    def train_on_epoch(self, data_loader):
        self.model.train()
        y_true, y_pred = [], []
        
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        for t_batch in data_loader:
            *x_batch, y_batch = [t.to(device=self.device) for t in t_batch]

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_batch_pred = self.model(*x_batch)
                loss = self.loss(y_batch_pred, y_batch)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()

            y_pred.append(y_batch_pred.detach())
            y_true.append(y_batch)

        y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
        return self.__calc_metrics(y_pred, y_true)

    @torch.no_grad()
    def evaluate(self, data_loader, use_amp):
        self.model.eval()
        y_true, y_pred = [], []
 
        for t_batch in data_loader:
            *x_batch, y_batch = [t.to(device=self.device) for t in t_batch]

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_batch_pred = self.model(*x_batch)

            y_pred.append(y_batch_pred) 
            y_true.append(y_batch)

        y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
        return self.__calc_metrics(y_pred, y_true)



class Logger(object):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, verbose, train_loader, val_loader):
        self.verbose = verbose
        if self.verbose == 0:
            return None
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
            self.ordered_metrics_keys = train_columns + val_columns
            self.history = pd.DataFrame(columns=self.ordered_metrics_keys)
        
        time_elapsed = time.time() - self.time

        content = []
        content.append(f"Epoch {kwargs['epoch']}/{kwargs['max_epochs']}")

        time_str = f'{round(time_elapsed, 1)}s' if time_elapsed < 9.5 else f'{int(time_elapsed + 0.5)}s'
        content.append(time_str)
        
        self.metrics = kwargs['train_metrics']
        self.metrics.update({('val_' + k): v for k, v in kwargs['val_metrics'].items()})
 
        for k in self.ordered_metrics_keys:
            content.append(f'{k}: ' + '{:.4f}'.format(self.metrics[k])) 

        self.metrics['lr'] = self.trainer.optimizer.param_groups[0]['lr']
        content.append('lr: {:.0e}'.format(self.metrics['lr']))

        self.history.loc[kwargs['epoch']] = self.metrics

        if self.verbose == 1:
            print(' - '.join(content))
        elif self.verbose == 2:
            print('|'.join(content).replace(' ', '').replace('Epoch', ''))