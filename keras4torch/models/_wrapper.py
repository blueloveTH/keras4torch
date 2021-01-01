import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Subset
from collections import OrderedDict
from .._summary import summary

from .._training import Trainer
from ..layers import KerasLayer
from ..metrics import _to_metrics_dic
from ..losses import _create_loss
from ..optimizers import _create_optimizer
from ..activations import _create_activation
from ..utils import to_tensor, _get_num_workers


class Model(torch.nn.Module):
    """
    `Model` wraps a `nn.Module` with training and inference features.

    Once the model is wrapped, you can config the model with losses and metrics\n  with `model.compile()`, train the model with `model.fit()`, or use the model\n  to do prediction with `model.predict()`.
    """
    def __init__(self, model):
        super(Model, self).__init__()
        self._k4t_model_tag = 0
        assert not hasattr(model, '_k4t_model_tag')

        self.model = model
        self.compiled = False
        self.built = False

        # set default configs
        self.shared_configs = {'batch_size': 32, 'num_workers': 0, 'use_amp': False,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'}

        def dfs(m):
            for child in m.children():
                if isinstance(child, KerasLayer) or dfs(child):
                    return True
            return False
        self._has_keras_layer = dfs(self)

    def forward(self, *args):
        return self.model(*args)

    def count_params(self) -> int:
        """Count the total number of scalars composing the weights."""
        return sum([p.numel() for p in self.parameters()])

    def trainable_params(self):
        """Return all trainable parameters of the model."""
        return filter(lambda p: p.requires_grad, self.parameters())


    def _check_keras_layer(self):
        if self._has_keras_layer and not self.built:
            raise AssertionError(
                "You need to build the model via `.build()` before this operation. Because it contains `KerasLayer`."
                )

    def _get_configs(self, save=False, **kwargs):
        configs = {}
        for k, v in kwargs.items():
            configs[k] = v if v != None else self.shared_configs[k]
        if save:
            self.shared_configs.update(configs)
    
        if len(configs) == 1:
            return list(configs.values())[0]
        else:
            return configs


    ########## keras-style methods below ##########

    @torch.no_grad()
    def build(self, input_shape):
        """Build the model when it contains `KerasLayer`."""
        if isinstance(input_shape[0], list) or isinstance(input_shape[0], tuple):
            batch_shapes = [ [2]+list(i) for i in input_shape]
            probe_inputs = [torch.zeros(size=s) for s in batch_shapes]
        else:
            batch_shape = [2] + list(input_shape)
            probe_inputs = [torch.zeros(size=batch_shape)]
        self.model(*probe_inputs)

        self.built = True
        self._probe_inputs = probe_inputs
        return self

    def summary(self, depth=3, device=None):
        """Print a string summary of the network."""
        self._check_keras_layer()
        if not self.built:
            raise AssertionError('Build the model first before you call `.summary()`.')

        device = self._get_configs(device=device)
        summary(self.model, self._probe_inputs, depth=depth, verbose=1, device=device)

    def compile(self, optimizer, loss, metrics=None, device=None):
        """
        Configure the model for training.

        Args:

        * `optimizer`: String (name of optimizer) or optimizer instance.

        * `loss`: String (name of objective function), objective function or loss instance.

        * `metrics`: List of metrics to be evaluated by the model during training. You can also use dict to specify the 
        abbreviation of each metric.

        * `device`: Device of the model and its trainer, if `None` 'cuda' will be used when `torch.cuda.is_available()` otherwise 'cpu'.
        """
        self._check_keras_layer()
        device = self._get_configs(save=True, device=device)
            
        loss = _create_loss(loss)
        optimizer = _create_optimizer(optimizer, self.trainable_params())

        m = OrderedDict({'loss': loss})
        m.update(_to_metrics_dic(metrics))

        self.to(device=device)
        self.trainer = Trainer(model=self, optimizer=optimizer, loss=loss, metrics=m, device=device)
        self.compiled = True


    def fit_dl(self, train_loader, epochs,
                val_loader=None,
                callbacks=[],
                verbose=1,
                use_amp=None):

        assert self.compiled
        self.trainer.register_callbacks(callbacks)

        use_amp = self._get_configs(save=True, use_amp=use_amp)
        history = self.trainer.run(train_loader, val_loader, max_epochs=epochs, verbose=verbose, use_amp=use_amp)

        return history


    def fit(self, x, y, epochs, batch_size=None,
                validation_split=None, shuffle_val_split=True,
                validation_data=None,
                callbacks=[],
                verbose=1,
                shuffle=True,
                sample_weight=None,
                num_workers=None,
                use_amp=None
                ):
        """
        Train the model for a fixed number of epochs (iterations on a dataset).

        Args:

        * `x` (`ndarray` or `torch.Tensor` or list): Input data

        * `y` (`ndarray` or `torch.Tensor`): Target data

        * `epochs` (int): Number of epochs to train the model

        * `batch_size` (int, default=32): Number of samples per gradient update

        * `validation_split` (float between 0 and 1): Fraction of the training data to be used as validation data

        * `shuffle_val_split` (bool, default=True): Whether to do shuffling when `validation_split` is provided

        * `validation_data` (tuple of `x` and `y`): Data on which to evaluate the loss and any model metrics at the end of each epoch
        
        * `callbacks` (list of `keras4torch.callbacks.Callback`): List of callbacks to apply during training

        * `verbose` (int, default=1): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = normal, 2 = brief

        * `shuffle` (bool, default=True): Whether to shuffle the training data before each epoch

        * `sample_weight` (list of floats): Optional weights for the training samples. If provided will enable `WeightedRandomSampler`
        
        * `num_workers` (int, default=0): Workers of `DataLoader`. If `-1` will use `cpu_count() - 1` for multiprocessing

        * `use_amp` (bool, default=False): Whether to use automatic mixed precision
        """

        assert self.compiled
        if not isinstance(x, list) and not isinstance(x, tuple):
            x = [x]
        x, y = to_tensor(x, y)

        assert not (validation_data != None and validation_split != None)
        has_val = validation_data != None or validation_split != None
        
        train_set = TensorDataset(*x, y)

        if validation_data != None:
            x_val, y_val = to_tensor(validation_data[0], validation_data[1])
            if not isinstance(x_val, list) and not isinstance(x_val, tuple):
                x_val = [x_val]
            val_set = TensorDataset(*x_val, y_val)

        if validation_split != None:
            val_length = int(len(train_set) * validation_split)
            idx = np.arange(0, len(train_set))
            if shuffle_val_split:
                np.random.seed(1234567890)
                np.random.shuffle(idx)
            train_set, val_set = Subset(train_set, idx[:-val_length]), Subset(train_set, idx[-val_length:])
            if sample_weight is not None:
                assert len(sample_weight) == len(y)
                sample_weight = [sample_weight[i] for i in idx[:-val_length]]

        if sample_weight is not None:
            sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
            shuffle = None
        else:
            sampler = None
        
        dl_configs = self._get_configs(save=True, batch_size=batch_size, num_workers=_get_num_workers(num_workers))
        train_loader = DataLoader(train_set, shuffle=shuffle, sampler=sampler, **dl_configs)
        val_loader = DataLoader(val_set, shuffle=False, **dl_configs) if has_val else None

        return self.fit_dl(train_loader, epochs, val_loader, callbacks, verbose, use_amp)

    @torch.no_grad()
    def evaluate(self, x, y, batch_size=None, num_workers=None, use_amp=None):
        """Return the loss value & metrics values for the model in test mode.\n\n    Computation is done in batches."""
        if not isinstance(x, list) and not isinstance(x, tuple):
            x = [x]
        x, y = to_tensor(x, y)

        dl_configs = self._get_dl_configs(copy=True, batch_size=batch_size, num_workers=_get_num_workers(num_workers))
        val_loader = DataLoader(TensorDataset(*x, y), shuffle=False, **dl_configs)
        return self.evaluate_dl(val_loader, use_amp)

    @torch.no_grad()
    def evaluate_dl(self, data_loader, use_amp=None):
        assert self.compiled
        use_amp = self._get_configs(use_amp=use_amp)
        return self.trainer.evaluate(data_loader, use_amp=use_amp)

    @torch.no_grad()
    def predict_dl(self, data_loader, device=None, output_numpy=True, activation=None, use_amp=None):
        self._check_keras_layer()

        device = self._get_configs(device=device)
        use_amp = self._get_configs(use_amp=use_amp)

        self.eval().to(device=device)

        outputs = []
        for batch in data_loader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs.append(self(*batch))

        outputs = torch.cat(outputs, dim=0)

        activation = _create_activation(activation)
        if activation != None:
            outputs = activation(outputs)

        return outputs.cpu().numpy() if output_numpy else outputs

    @torch.no_grad()
    def predict(self, x, batch_size=None, device=None, output_numpy=True, activation=None, num_workers=None, use_amp=None):
        """Generate output predictions for the input samples.\n\n    Computation is done in batches."""
        if not isinstance(x, list) and not isinstance(x, tuple):
            x = [x]
        dl_configs = self._get_configs(batch_size=batch_size, num_workers=_get_num_workers(num_workers))
        data_loader = DataLoader(TensorDataset(*to_tensor(x)), shuffle=False, **dl_configs)
        return self.predict_dl(data_loader, device=device, output_numpy=output_numpy, activation=activation, use_amp=use_amp)

    def save_weights(self, filepath):
        """Equal to `torch.save(model.state_dict(), filepath)`."""
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        """Equal to `model.load_state_dict(torch.load(filepath))`."""
        self.load_state_dict(torch.load(filepath))
