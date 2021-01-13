import torch
import time
import pandas as pd
from enum import Enum
from collections import OrderedDict
from .utils._keras_progress_bar import Progbar

from copy import deepcopy

from ._custom_training import _default_trainer_loop_config

class Events(Enum):
    ON_EPOCH_END = 'on_epoch_end'
    ON_EPOCH_BEGIN = 'on_epoch_begin'
    ON_TRAIN_BEGIN = 'on_train_begin'
    ON_TRAIN_END = 'on_train_end'


class StopTrainingError(Exception):
    pass

@torch.no_grad()
def calc_metrics(y_pred, y_true, metrics_dic: OrderedDict):
    metrics = OrderedDict()
    for key, score_fn in metrics_dic.items():
        metrics[key] = score_fn(y_pred, y_true).mean().cpu().item()
    return metrics

class MetricsRecorder():
    def __init__(self, metrics, epoch_metrics) -> None:
        self.metrics = metrics
        self.epoch_metrics = OrderedDict([(k+'#', v) for k,v in epoch_metrics.items()])
        self.has_epoch_metrics = len(self.epoch_metrics) > 0

        # for batch metrics
        self.accum_metrics = OrderedDict([(k, 0.0) for k in self.metrics.keys()])
        self.total_count = 0

        # for epoch metrics
        self.y_pred = []
        self.y_true = []

    def update(self, y_batch_pred, y_batch):
        self.__update_batch_metrics(y_batch_pred, y_batch)
        if self.has_epoch_metrics:
            self.y_pred.append(y_batch_pred.cpu())
            self.y_true.append(y_batch.cpu())

    def __update_batch_metrics(self, y_batch_pred, y_batch):
        count = len(y_batch)
        if count == 0:
            return
        batch_metrics = calc_metrics(y_batch_pred, y_batch, self.metrics)
        
        for key, value in batch_metrics.items():
            self.accum_metrics[key] += value * count
        self.total_count += count

    def average_epoch_metrics(self, free_memory):
        if not self.has_epoch_metrics:
            return OrderedDict()
        t_0 = torch.cat(self.y_pred)
        t_1 = torch.cat(self.y_true)
        if free_memory:
            del self.y_pred, self.y_true
        return calc_metrics(t_0, t_1, self.epoch_metrics)

    def average_batch_metrics(self):
        return OrderedDict([(k, (v/self.total_count)) for k,v in self.accum_metrics.items()])

    def average(self, free_memory=True):
        metrics = self.average_batch_metrics()
        metrics.update(self.average_epoch_metrics(free_memory))
        return metrics


class Trainer(object):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.loop_config is None:
            self.loop_config = _default_trainer_loop_config
    
    def register_callbacks(self, callbacks):
        self.event_dict = {Events(k).value: [] for k in Events}

        if callbacks is None:
            return

        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for c in callbacks:
            for key, func in c.get_callbacks_dict().items():
                self.event_dict[Events(key).value] += [func]
    
    def __fire_event(self, key):
        func_list = self.event_dict[Events(key).value]
        for func in func_list:
            func(self)

    def get_loop_config(self, train):
        loop = deepcopy(self.loop_config)
        loop.train = train
        return loop

    def run(self, train_loader, val_loader, max_epochs, verbose, use_amp, accum_grad_steps):
        self.data_loaders = train_loader, val_loader
        self.max_epochs = max_epochs
        self.logger = Logger(self, verbose=verbose)
        self.use_amp = use_amp if self.device == 'cuda' else False
        self.accum_grad_steps = accum_grad_steps
        
        self.logger.on_train_begin(train_loader, val_loader)
        self.__fire_event(Events.ON_TRAIN_BEGIN)

        for epoch in range(1, max_epochs+1):
            self.epoch = epoch
            self.__fire_event(Events.ON_EPOCH_BEGIN)

            self.logger.on_epoch_begin(epoch, max_epochs, data_loader=train_loader)

            train_metrics = self.train_on_epoch(train_loader)
            val_metrics = self.valid_on_epoch(val_loader, self.use_amp) if val_loader else OrderedDict()

            self.logger.on_epoch_end(epoch, train_metrics, val_metrics)

            try:
                self.__fire_event(Events.ON_EPOCH_END)
            except StopTrainingError:
                break

        self.__fire_event(Events.ON_TRAIN_END)
        return self.logger.history

    def train_on_epoch(self, data_loader):
        self.model.train()
        metrics_rec = MetricsRecorder(self.metrics, self.epoch_metrics)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        loop = self.get_loop_config(train=True)

        self.optimizer.zero_grad()

        batch_idx, max_batch_idx = 0, len(data_loader)
        for batch in data_loader:
            *x_batch, y_batch = [i.to(device=self.device) for i in batch]
            x_batch, y_batch = loop.process_batch(x_batch, y_batch)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_batch_pred = loop.forward_call(self.model, x_batch)
                loss = self.loss(y_batch_pred, y_batch)
                loss = loss / self.accum_grad_steps

            grad_scaler.scale(loss).backward()

            # https://discuss.pytorch.org/t/pytorch-gradient-accumulation/55955

            _is_accum_end = (batch_idx + 1) % self.accum_grad_steps == 0
            _is_last_batch = (batch_idx + 1) == max_batch_idx

            if _is_accum_end or _is_last_batch:
                loop.prepare_for_optimizer_step(self.model)
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad()

            y_batch_pred = y_batch_pred.detach().float()
            metrics_rec.update(
                *loop.prepare_for_metrics_update(y_batch_pred, y_batch)
            )

            self.logger.on_batch_end(metrics_rec)
            batch_idx += 1

        return metrics_rec.average()

    

    @torch.no_grad()
    def valid_on_epoch(self, data_loader, use_amp):
        self.model.eval()
        metrics_rec = MetricsRecorder(self.metrics, self.epoch_metrics)

        loop = self.get_loop_config(train=False)
 
        for batch in data_loader:
            *x_batch, y_batch = [i.to(device=self.device) for i in batch]
            x_batch, y_batch = loop.process_batch(x_batch, y_batch)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_batch_pred = loop.forward_call(self.model, x_batch)

            y_batch_pred = y_batch_pred.detach().float()
            metrics_rec.update(
                *loop.prepare_for_metrics_update(y_batch_pred, y_batch)
            )

        return metrics_rec.average()


class Logger(object):
    def __init__(self, trainer, verbose):
        self.trainer = trainer
        self.verbose = verbose

    def on_train_begin(self, train_loader, val_loader):
        if self.verbose == 0:
            return None
        if val_loader is not None:
            print(f'Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples:')
        else:
            print(f'Train on {len(train_loader.dataset)} samples:')

    def on_epoch_begin(self, epoch, max_epochs, data_loader):
        self.time = time.time()

        if self.verbose == 2:
            print(f"Epoch {epoch}/{max_epochs}", end='')
        if self.verbose == 1:
            print(f"Epoch {epoch}/{max_epochs}")    # new line
            self.bar = Progbar(len(data_loader))
            self.step_count = 0

    def on_batch_end(self, metrics_rec: MetricsRecorder):
        if self.verbose == 1:
            self.step_count += 1
            self.bar.update(self.step_count, metrics_rec.average_batch_metrics(), finalize=False)
            
    def on_epoch_end(self, epoch, train_metrics: OrderedDict, val_metrics: OrderedDict):        
        time_elapsed = time.time() - self.time
        time_str = f'{round(time_elapsed, 1)}s' if time_elapsed < 9.5 else f'{int(time_elapsed + 0.5)}s'

        content = ['', time_str]
        
        self.metrics = train_metrics
        self.metrics.update({('val_' + k): v for k, v in val_metrics.items()})
 
        for k, v in self.metrics.items():
            if v > 1e-3:
                content.append('{}: {:.4f}'.format(k, v))
            else:
                content.append('{}: {:.4e}'.format(k, v))

        self.metrics['lr'] = self.trainer.optimizer.param_groups[0]['lr']
        content.append('lr: {:.1e}'.format(self.metrics['lr']).replace('.0e', 'e'))

        if not hasattr(self, 'history'):
            self.history = pd.DataFrame(columns=list(self.metrics.keys()))
        self.history.loc[epoch] = self.metrics

        if self.verbose > 0:
            if self.verbose == 1:
                self.bar.update(self.step_count, {}, finalize=True)
            print(' - '.join(content), flush=True)