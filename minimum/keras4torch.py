import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import time
from collections import OrderedDict
import pandas as pd
import numpy as np

__version__ = '1.0.0-mini'

def to_tensor(*args):
    rt = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = torch.from_numpy(arg)
        elif not isinstance(arg, torch.Tensor):
            raise TypeError('Only np.ndarray and torch.Tensor are supported.')

        if arg.dtype == torch.float64:
            print('[Warning] Auto convert float64 to float32, this could lead to extra memory usage.')
            arg = arg.float()
        elif arg.dtype == torch.int32:
            print('[Warning] Auto convert int32 to int64, this could lead to extra memory usage.')
            arg = arg.long()

        rt.append(arg)
                
    return rt[0] if len(rt) == 1 else tuple(rt)


class Model(torch.nn.Module):
    """
    `Model` wraps a `nn.Module` with training and inference features.

    Once the model is wrapped, you can config the model with losses and metrics\n  with `model.compile()`, train the model with `model.fit()`, or use the model\n  to do prediction with `model.predict()`.
    """
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.compiled = False

    def forward(self, x):
        return self.model(x)

    def count_params(self) -> int:
        """Count the total number of scalars composing the weights."""
        return sum([p.numel() for p in self.parameters()])

    def compile(self, optimizer, loss, metrics=None, device=None):
        """
        Configure the model for training.

        Args:

        * `optimizer`: pytorch optimizer instance.

        * `loss`: loss instance or `fn(y_pred, y_true)`.

        * `metrics`: List of `fn(y_pred, y_true)` to be evaluated by the model during training. You can also use dict to specify the 
        abbreviation of each metric.

        * `device`: Device of the model and its trainer, if `None` 'cuda' will be used when `torch.cuda.is_available()` otherwise 'cpu'.
        """
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        m = OrderedDict({'loss': loss})
        if isinstance(metrics, dict):
            m.update(metrics)
        elif isinstance(metrics, list):
            m.update({tmp_m.__name__: tmp_m for tmp_m in metrics if callable(tmp_m)})
        elif not (metrics is None):
            raise TypeError('Argument `metrics` should be either a dict or list.')

        self.to(device=device)
        self.trainer = Trainer(model=self, optimizer=optimizer, loss=loss, metrics=m, device=device)
        self.compiled = True

    ###### DataLoader workflow ######

    def fit_dl(self, train_loader, epochs,
                val_loader=None,
                verbose=1):

        assert self.compiled
        return self.trainer.run(train_loader, val_loader, max_epochs=epochs, verbose=verbose)

    @torch.no_grad()
    def evaluate_dl(self, data_loader):
        assert self.compiled
        return self.trainer.evaluate(data_loader)

    @torch.no_grad()
    def predict_dl(self, data_loader, device=None, output_numpy=True, activation=None):
        if device == None:
            if self.compiled:
                device = self.trainer.device
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval().to(device=device)

        outputs = [self(x_batch[0].to(device=device)) for x_batch in data_loader]
        outputs = torch.cat(outputs, dim=0)

        if activation != None:
            outputs = activation(outputs)

        return outputs.cpu().numpy() if output_numpy else outputs

    ###### NumPy workflow ######

    def fit(self, x, y, epochs, batch_size=32,
                validation_split=None, shuffle_val_split=True,
                validation_data=None,
                verbose=1,
                shuffle=True,
                ):
        """
        Train the model for a fixed number of epochs (iterations on a dataset).

        Args:

        * `x` (`ndarray` or `torch.Tensor`): Input data 

        * `y` (`ndarray` or `torch.Tensor`): Target data

        * `epochs` (int): Number of epochs to train the model

        * `batch_size` (int, default=32): Number of samples per gradient update

        * `validation_split` (float between 0 and 1): Fraction of the training data to be used as validation data

        * `shuffle_val_split` (bool, default=True): Whether to do shuffling when `validation_split` is provided

        * `validation_data` (tuple of `x` and `y`): Data on which to evaluate the loss and any model metrics at the end of each epoch
        
        * `verbose` (int, default=1): 0, 1, or 2. Verbosity mode. 0 = disabled, 1 = enabled

        * `shuffle` (bool, default=True): Whether to shuffle the training data before each epoch
        """

        assert self.compiled
        x, y = to_tensor(x, y)

        assert not (validation_data != None and validation_split != None)
        has_val = validation_data != None or validation_split != None
        
        train_set = TensorDataset(x, y)

        if validation_data != None:
            x_val, y_val = to_tensor(validation_data[0], validation_data[1])
            val_set = TensorDataset(x_val, y_val)

        if validation_split != None:
            val_length = int(len(train_set) * validation_split)
            idx = np.arange(0, len(train_set))
            if shuffle_val_split:
                np.random.seed(1234567890)
                np.random.shuffle(idx)
            train_set, val_set = Subset(train_set, idx[:-val_length]), Subset(train_set, idx[-val_length:])
        
        train_loader = DataLoader(train_set, shuffle=shuffle, batch_size=batch_size)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size) if has_val else None

        return self.fit_dl(train_loader, epochs, val_loader, verbose)

    @torch.no_grad()
    def evaluate(self, x, y, batch_size=32):
        """Return the loss value & metrics values for the model in test mode.\n\n    Computation is done in batches."""
        x, y = to_tensor(x, y)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
        return self.evaluate_dl(val_loader)

    @torch.no_grad()
    def predict(self, inputs, batch_size=32, device=None, output_numpy=True, activation=None):
        """Generate output predictions for the input samples.\n\n    Computation is done in batches."""
        inputs = to_tensor(inputs)
        data_loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)
        return self.predict_dl(data_loader, device=device, output_numpy=output_numpy, activation=activation)

    ###### Save & Load ######

    def save_weights(self, filepath):
        """Equal to `torch.save(model.state_dict(), filepath)`."""
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        """Equal to `model.load_state_dict(torch.load(filepath))`."""
        self.load_state_dict(torch.load(filepath))


class Trainer(object):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = Logger(self)
    
    def run(self, train_loader, val_loader, max_epochs, verbose):
        self.max_epochs = max_epochs
        self.logger.on_train_begin(verbose, train_loader, val_loader)

        for epoch in range(1, max_epochs+1):
            self.epoch = epoch
            self.logger.on_epoch_begin()

            train_metrics = self.train_fast_mode(train_loader)
            val_metrics = self.evaluate(val_loader) if val_loader else {}

            self.logger.on_epoch_end(epoch=epoch, max_epochs=max_epochs, train_metrics=train_metrics, val_metrics=val_metrics)
        return self.logger.history

    @torch.no_grad()
    def __calc_metrics(self, y_pred, y_true):
        metrics = {}
        for key, score_fn in self.metrics.items():
            metrics[key] = score_fn(y_pred, y_true).mean().cpu().item()
        return metrics

    def train_fast_mode(self, data_loader):
        self.model.train()
        y_true, y_pred = [], []
        
        for t in data_loader:
            x, y = t[0].to(device=self.device), t[1].to(device=self.device)
            self.optimizer.zero_grad()

            y_batch_pred = self.model(x)
            self.loss(y_batch_pred, y).backward()
            self.optimizer.step()

            y_pred.append(y_batch_pred.detach())
            y_true.append(y)

        return self.__calc_metrics(torch.cat(y_pred), torch.cat(y_true))

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        y_true, y_pred = [], []
 
        for t in data_loader:
            x, y = t[0].to(device=self.device), t[1].to(device=self.device)
            y_pred.append(self.model(x)) 
            y_true.append(y)

        return self.__calc_metrics(torch.cat(y_pred), torch.cat(y_true))


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
            train_columns = list(kwargs['train_metrics'].keys())
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