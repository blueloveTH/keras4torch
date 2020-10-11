import torch
from torch._C import device
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torchsummary
from collections import OrderedDict
from .callbacks import *
from enum import Enum

class Events(Enum):
    ON_EPOCH_END = 'on_epoch_end'
    ON_EPOCH_START = 'on_epoch_start'

class _Trainer(object):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = _Logger(self)
    
    def register_callbacks(self, callbacks):
        self.event_dict = {Events(k).value: list() for k in Events}
        for key, func in callbacks.items():
            self.event_dict[Events(key).value] += [func]
    
    def __fire_event(self, key):
        func_list = self.event_dict[Events(key).value]
        for func in func_list:
            func(self)

    def run(self, train_loader, val_loader, max_epochs, verbose):
        self.max_epochs = max_epochs
        self.logger.verbose = verbose
        self.logger.on_train_start(train_loader, val_loader)

        for epoch in range(1, max_epochs+1):
            self.epoch = epoch
            self.logger.on_epoch_start()
            self.__fire_event(Events.ON_EPOCH_START)

            for x_batch, y_batch in train_loader:
                self.train_step(x_batch, y_batch)

            train_metrics = self.evaluate(train_loader)
            val_metrics = self.evaluate(val_loader) if val_loader else {}

            self.logger.on_epoch_end(epoch=epoch, max_epochs=max_epochs, train_metrics=train_metrics, val_metrics=val_metrics)
            self.__fire_event(Events.ON_EPOCH_END)

        return self.logger.history
            
    def train_step(self, x, y):
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        self.model.train()    # Sets the model in training mode
        self.optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = self.model.forward(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return None

    @torch.no_grad()
    def evaluate_step(self, x, y):
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        m = torch.zeros(size=[len(self.metrics)])
        self.model.eval()
        y_pred = self.model.forward(x)
        for i, score_func in enumerate(self.metrics.values()):
            m[i] = score_func(y_pred, y)
        return m

    @torch.no_grad()
    def evaluate(self, data_loader):
        metrics = []
        for x_batch, y_batch in data_loader:
            metrics.append(self.evaluate_step(x_batch, y_batch))
        metrics = torch.stack(metrics).mean(dim=0).cpu().numpy()
        return OrderedDict({k:v for k,v in zip(self.metrics.keys(), metrics)})

class _Logger(object):
    def __init__(self, trainer):
        self.trainer = trainer
        self.verbose = 0

    def on_train_start(self, train_loader, val_loader):
        if val_loader != None:
            print(f'Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples:')
        else:
            print(f'Train on {len(train_loader.dataset)} samples:')

    def on_epoch_start(self, **kwargs):
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



class Model(torch.nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.compiled = False

    def forward(self, x):
        return self.model.forward(x)

    @staticmethod
    def to_tensor(*args):
        rt = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                arg = torch.from_numpy(arg.values)
            elif isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
            elif not isinstance(arg, torch.Tensor):
                raise TypeError('Only DataFrame, ndarray and torch.Tensor are supported.')
            rt.append(arg)
                
        return rt[0] if len(rt) == 1 else tuple(rt)

    def get_params_cnt(self):
        return sum([p.numel() for p in self.model.parameters()])

    ########## keras-like methods below ##########

    def summary(self, input_shape, depth=3, verbose=1):
        torchsummary.summary(self.model, input_shape, depth=depth, verbose=verbose)

    def compile(self, optimizer, loss, device=None, metrics={}):
        self.compiled = True
        m = OrderedDict({'loss': loss})
        m.update(metrics)
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device=device)
        self.trainer = _Trainer(model=self, optimizer=optimizer, loss=loss, metrics=m, device=device)


    def fit(self, x, y, epochs, batch_size=32,
                validation_split=None, shuffle_val_split=False,
                validation_data=None,
                callbacks={}, 
                verbose=1
                ):

        assert self.compiled
        x, y = self.to_tensor(x, y)

        assert not (validation_data != None and validation_split != None)
        has_val = validation_data != None or validation_split != None

        if validation_split != None:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, shuffle=shuffle_val_split)
        else:
            x_train = x; y_train = y
            if validation_data != None:
                x_val, y_val = self.to_tensor(validation_data[0], validation_data[1])

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False) if has_val else None

        # Training
        self.trainer.register_callbacks(callbacks)
        history = self.trainer.run(train_loader, val_loader, max_epochs=epochs, verbose=verbose)

        return history

    @torch.no_grad()
    def evaluate(self, x, y, batch_size=32):
        assert self.compiled
        x, y = self.to_tensor(x, y)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
        return self.trainer.evaluate(val_loader)

    @torch.no_grad()
    def predict(self, inputs, batch_size=32, device=None):
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        inputs = self.to_tensor(inputs)
        outputs = []
        self.model.eval()

        data_loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)
        for x_batch in data_loader:
            outputs.append(self.model.forward(x_batch[0].to(device=device)))

        return torch.cat(outputs, dim=0).cpu().numpy()
