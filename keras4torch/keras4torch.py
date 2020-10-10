import torch
import ignite
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from torch.utils.data import DataLoader, TensorDataset
import abc
import time
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torchsummary


class Model(object):
    def __init__(self, model, device):
        self.model = model.to(device=device)
        self.device = device

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def parameters(self):
        return self.model.parameters()

    def summary(self, input_shape):
        torchsummary.summary(self.model, input_shape)

    def to(self, device):
        self.model.to(device=device)
        self.device = device
        return self

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            return self.model(self.to_tensor(inputs))

    def compile(self, optimizer, loss, metrics={}):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = {'loss': ignite.metrics.Loss(loss)}; self.metrics.update(metrics)

    def to_tensor(self, *args):
        rt = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                rt.append(torch.from_numpy(arg))
            else:
                rt.append(arg)
        if len(rt) == 1:
            return rt[0]
        else:
            return tuple(rt)

    def fit(self, x, y, batch_size, epochs, validation_split=None, validation_data=None, callbacks={}, verbose=1):
        x, y = self.to_tensor(x, y)

        assert not (validation_data != None and validation_split != None)
        has_val = validation_data != None or validation_split != None

        if validation_split != None:
            x_train, x_val, y_train, y_val = train_test_split(x, y,
                                        test_size=validation_split, shuffle=False)
            print(f'Train on {x_train.shape[0]} samples, validate on {x_val.shape[0]} samples:')
        else:
            x_train = x
            y_train = y

            if validation_data != None:
                x_val, y_val = self.to_tensor(validation_data[0], validation_data[1])
                print(f'Train on {x_train.shape[0]} samples, validate on {x_val.shape[0]} samples:')
            else:
                print(f'Train on {x_train.shape[0]} samples:')

        trainer = create_supervised_trainer(
            self.model, self.optimizer, self.loss, device=self.device)

        evaluator = create_supervised_evaluator(
            self.model, self.metrics, device=self.device)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        if has_val:
            val_loader = DataLoader(
                TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
        else:
            val_loader = None

        self.logger = Logger(verbose, trainer, evaluator, train_loader, val_loader, self.optimizer)
        trainer.model_wrapper = self

        for e, handler in callbacks.items():
            trainer.add_event_handler(e, handler, {'model': self.model})
        
        trainer.run(train_loader, max_epochs=epochs)
        return pd.concat(self.logger.history)

    def evaluate(self, x, y, batch_size):
        x, y = self.to_tensor(x, y)

        evaluator = create_supervised_evaluator(
            self.model, self.metrics, device=self.device)

        val_loader = DataLoader(
            TensorDataset(x, y), batch_size=batch_size, shuffle=False)

        evaluator.run(val_loader)
        return evaluator.state.metrics
        
class Logger(object):
    def __init__(self, verbose, trainer, evaluator, train_loader, val_loader, optimizer):
        self.verbose = verbose
        self.trainer = trainer
        self.evaluator = evaluator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        trainer.add_event_handler(Events.EPOCH_STARTED, self.on_epoch_begin)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_end)
        trainer.add_event_handler(Events.COMPLETED, self.on_train_end)

        self.history = []

    def on_epoch_begin(self, trainer):
        self.time = time.time()

    def on_epoch_end(self, trainer):
        time_elapsed = time.time() - self.time
        content = []
        content.append(f'Epoch {trainer.state.epoch}/{trainer.state.max_epochs}')

        if time_elapsed < 10:
            content.append(f'{round(time_elapsed, 1)}s')
        else:
            content.append(f'{int(time_elapsed + 0.5)}s')
        
        epoch_rec = {}
        columns = []

        self.evaluator.run(self.train_loader)
        for k, v in self.evaluator.state.metrics.items():
            content.append(f'{k}: ' + '{:.4f}'.format(v))
            columns.append(k)
            epoch_rec[k] = v

        if self.val_loader != None:
            self.evaluator.run(self.val_loader)
            for k, v in self.evaluator.state.metrics.items():
                k = 'val_' + k
                content.append(f'{k}: ' + '{:.4f}'.format(v))
                columns.append(k)
                epoch_rec[k] = v

        epoch_rec['lr'] = self.optimizer.param_groups[0]['lr']
        content.append('lr: {:.0e}'.format(epoch_rec['lr']))

        self.history.append(pd.DataFrame(epoch_rec,
                    columns=columns, index=[trainer.state.epoch]))
        
        self.metrics = epoch_rec
        if self.verbose == 1:
            print(' - '.join(content))
        elif self.verbose == 2:
            print('|'.join(content).replace(' ', '').replace('Epoch', ''))


