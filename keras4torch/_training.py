import torch
import time
import pandas as pd
from enum import Enum
from .utils import Progbar

class Events(Enum):
    ON_EPOCH_END = 'on_epoch_end'
    ON_EPOCH_BEGIN = 'on_epoch_begin'
    ON_TRAIN_BEGIN = 'on_train_begin'
    ON_TRAIN_END = 'on_train_end'


class StopTrainingError(Exception):
    def __init__(self):
        pass

class MetricsRecorder():
    def __init__(self, keys) -> None:
        self.metrics = {k: 0.0 for k in keys}
        self.total_count = 0

    def add(self, batch_metrics, count):
        for key, value in batch_metrics.items():
            self.metrics[key] += value * count
        self.total_count += count

    def average(self):
        return {k:(v/self.total_count) for k,v in self.metrics.items()}


class Trainer(object):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
    
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
        self.logger = Logger(self, verbose=verbose)

        self.data_loaders = train_loader, val_loader

        self.use_amp = use_amp if self.device == 'cuda' else False
        self.max_epochs = max_epochs
        self.logger.on_train_begin(train_loader, val_loader)
        self.__fire_event(Events.ON_TRAIN_BEGIN)

        for epoch in range(1, max_epochs+1):
            self.epoch = epoch
            self.__fire_event(Events.ON_EPOCH_BEGIN)

            self.logger.on_epoch_begin(epoch=epoch, max_epochs=max_epochs, data_loader=train_loader)

            train_metrics = self.train_on_epoch(train_loader)
            val_metrics = self.evaluate(val_loader) if val_loader else {}

            self.logger.on_epoch_end(epoch=epoch, max_epochs=max_epochs, train_metrics=train_metrics, val_metrics=val_metrics)

            try:
                self.__fire_event(Events.ON_EPOCH_END)
            except StopTrainingError:
                break

        self.__fire_event(Events.ON_TRAIN_END)
        return self.logger.history

    @torch.no_grad()
    @staticmethod
    def __calc_metrics(y_pred, y_true, metrics_dic):
        metrics = {}
        for key, score_fn in metrics_dic.items():
            metrics[key] = score_fn(y_pred, y_true).mean().cpu().item()
        return metrics

    def train_on_epoch(self, data_loader):
        self.model.train()
        metrics_recorder = MetricsRecorder(self.metrics.keys())
        
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        for batch in data_loader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device=self.device)
            *x_batch, y_batch = batch

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_batch_pred = self.model(*x_batch)
                loss = self.loss(y_batch_pred, y_batch)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(self.optimizer)
            grad_scaler.update()

            y_batch_pred = y_batch_pred.detach()

            batch_metrics = self.__calc_metrics(y_batch_pred, y_batch, self.metrics)
            metrics_recorder.add(batch_metrics, len(y_batch))

            self.logger.on_batch_end()

        return metrics_recorder.average()

    @torch.no_grad()
    def evaluate(self, data_loader, use_amp=None):
        if use_amp is None:
            use_amp = self.use_amp
        self.model.eval()
        metrics_recorder = MetricsRecorder(self.metrics.keys())
 
        for batch in data_loader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device=self.device)
            *x_batch, y_batch = batch

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_batch_pred = self.model(*x_batch)

            y_batch_pred = y_batch_pred.detach()

            batch_metrics = self.__calc_metrics(y_batch_pred, y_batch, self.metrics)
            metrics_recorder.add(batch_metrics, len(y_batch))

        return metrics_recorder.average()

    @torch.no_grad()
    def evaluate_cpu(self, data_loader, metrics_dic, use_amp=None):
        if use_amp is None:
            use_amp = self.use_amp
        self.model.eval()
        y_pred, y_true = [], []
 
        for batch in data_loader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device=self.device)
            *x_batch, y_batch = batch

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_batch_pred = self.model(*x_batch)

            y_batch_pred = y_batch_pred.detach()

            y_pred.append(y_batch_pred.cpu())
            y_true.append(y_batch.cpu())

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)

        return self.__calc_metrics(y_pred, y_true, metrics_dic)


class Logger(object):
    def __init__(self, trainer, verbose):
        self.trainer = trainer
        self.verbose = verbose

    def on_train_begin(self, train_loader, val_loader):
        if self.verbose == 0:
            return None
        if val_loader != None:
            print(f'Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples:')
        else:
            print(f'Train on {len(train_loader.dataset)} samples:')

    def on_epoch_begin(self, **kwargs):
        self.time = time.time()

        if self.verbose == 2:
            print(f"Epoch {kwargs['epoch']}/{kwargs['max_epochs']}", end='')
        if self.verbose == 1:
            print(f"Epoch {kwargs['epoch']}/{kwargs['max_epochs']}")    # new line
            self.bar = Progbar(len(kwargs['data_loader']))
            self.step_count = 0

    def on_batch_end(self):
        if self.verbose == 1:
            self.step_count += 1
            self.bar.update(self.step_count)
            
    def on_epoch_end(self, **kwargs):
        if not hasattr(self, 'history'):
            train_columns = [key for key in kwargs['train_metrics'].keys()]
            val_columns = ['val_'+key for key in kwargs['val_metrics'].keys()]
            self.ordered_metrics_keys = train_columns + val_columns
            self.history = pd.DataFrame(columns=self.ordered_metrics_keys)
        
        time_elapsed = time.time() - self.time

        content = ['']
        time_str = f'{round(time_elapsed, 1)}s' if time_elapsed < 9.5 else f'{int(time_elapsed + 0.5)}s'
        content.append(time_str)
        
        self.metrics = kwargs['train_metrics']
        self.metrics.update({('val_' + k): v for k, v in kwargs['val_metrics'].items()})
 
        for k in self.ordered_metrics_keys:
            content.append(f'{k}: ' + '{:.4f}'.format(self.metrics[k])) 

        self.metrics['lr'] = self.trainer.optimizer.param_groups[0]['lr']
        content.append('lr: {:.0e}'.format(self.metrics['lr']))

        self.history.loc[kwargs['epoch']] = self.metrics

        if self.verbose > 0:
            print(' - '.join(content))
        #elif self.verbose == 3:
        #    print('|'.join(content).replace(' ', '').replace('Epoch', ''))