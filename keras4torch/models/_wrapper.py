import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Subset
from collections import OrderedDict

from torch.utils.data.dataset import Dataset
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
        self.input_shape = None

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

    @torch.no_grad()
    def build(self, input_shape, dtype=torch.float32, batch_size=8):
        """Build the model when it contains `KerasLayer`."""
        if isinstance(input_shape[0], int):
            input_shape = [input_shape]

        if not isinstance(dtype, list):
            dtype = [dtype] * len(input_shape)

        assert len(dtype) == len(input_shape)

        device = self.trainer.device if self.compiled else 'cpu'

        batch_shapes = [ [batch_size]+list(i) for i in input_shape]
        probe_inputs = [torch.zeros(size=sz).to(dtype=dt, device=device) for sz,dt in zip(batch_shapes, dtype)]
        self.model(*probe_inputs)

        self._probe_inputs = probe_inputs
        self.built = True
        return self

    def _check_keras_layer(self):
        if self._has_keras_layer and not self.built:
            raise AssertionError("You need to build the model via `.build()` before this operation. Because it contains `KerasLayer`.")

    def summary(self, depth=3, device=None):
        """Print a string summary of the network."""
        self._check_keras_layer()
        if not self.built:
            raise AssertionError('Build the model first before you call `.summary()`.')

        if device is None:
            device = self.trainer.device if self.compiled else 'cpu'
        
        summary(self.model, self._probe_inputs, depth=depth, verbose=1, device=device)

    def compile(self, optimizer, loss, metrics=None, epoch_metrics=None, device=None, loop_config=None):
        """
        Configure the model for training.

        Args:

        * `optimizer`: String (name of optimizer) or optimizer instance.

        * `loss`: String (name of objective function), objective function or loss instance.

        * `metrics`: List of metrics to be evaluated by the model during training. You can also use dict to specify the 
        abbreviation of each metric.

        * `epoch_metrics`: List of non-linear metrics(e.g. ROC_AUC) that need to be evaluated on epoch end.

        * `device`: Device of the model and its trainer, if `None` 'cuda' will be used when `torch.cuda.is_available()` otherwise 'cpu'.
        
        * `loop_config`: Optional `TrainerLoopConfig` object to customize training and validation loop
        """
        self._check_keras_layer()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        loss = _create_loss(loss)
        optimizer = _create_optimizer(optimizer, self.trainable_params())

        batch_metrics = OrderedDict({'loss': loss})
        batch_metrics.update(_to_metrics_dic(metrics))

        epoch_metrics = _to_metrics_dic(epoch_metrics)

        self.to(device=device)
        self.trainer = Trainer(model=self, optimizer=optimizer, loss=loss, metrics=batch_metrics, epoch_metrics=epoch_metrics, device=device, loop_config=loop_config)
        self.compiled = True


    def fit_dl(self, train_loader, val_loader=None,
                epochs=10,
                callbacks=None,
                verbose=1,
                use_amp=False,
                accum_grad_steps=1):

        assert self.compiled
        self.trainer.register_callbacks(callbacks)
        history = self.trainer.run(train_loader, val_loader, max_epochs=epochs, verbose=verbose, use_amp=use_amp, accum_grad_steps=accum_grad_steps)

        return history


    def fit(self, x, y=None, epochs=10, batch_size=32,
                validation_batch_size=None,
                validation_split=None, shuffle_val_split=True,
                validation_data=None,
                callbacks=None,
                verbose=1,
                shuffle=True,
                sample_weight=None,
                num_workers=0,
                use_amp=False,
                accum_grad_steps=1
                ):
        """
        Train the model for a fixed number of epochs (iterations on a dataset).

        Args:

        * `x` (`ndarray` or `torch.Tensor` or `Dataset`): Input data

        * `y` (`ndarray` or `torch.Tensor`): Target data

        * `epochs` (int, default=10): Number of epochs to train the model

        * `batch_size` (int, default=32): Number of samples per gradient update

        * `validation_batch_size` (int, default=None): Number of samples for each step on validation loop, if `None` will use `batch_size`

        * `validation_split` (float between 0 and 1): Fraction of the training data to be used as validation data

        * `shuffle_val_split` (bool, default=True): Whether to do shuffling when `validation_split` is provided

        * `validation_data` (tuple of `x` and `y` or `Dataset`): Data on which to evaluate the loss and any model metrics at the end of each epoch
        
        * `callbacks` (list of `keras4torch.callbacks.Callback`): List of callbacks to apply during training

        * `verbose` (int, default=1): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = normal, 2 = brief

        * `shuffle` (bool, default=True): Whether to shuffle the training data before each epoch

        * `sample_weight` (list of floats): Optional weights for the training samples. If provided will enable `WeightedRandomSampler`
        
        * `num_workers` (int, default=0): Workers of `DataLoader`. If `-1` will use `cpu_count() - 1` for multiprocessing

        * `use_amp` (bool, default=False): Whether to use automatic mixed precision

        * `accum_grad_steps` (int, default=1): How many steps to update the model parameters
        """

        assert self.compiled
        assert not (validation_data is not None and validation_split is not None)
        has_val = validation_data is not None or validation_split is not None

        if isinstance(x, Dataset):
            train_set = x
        else:
            if not isinstance(x, list) and not isinstance(x, tuple):
                x = [x]
            x, y = to_tensor(x, y)
            train_set = TensorDataset(*x, y)
        
        del x, y    # for preventing bugs

        if validation_data is not None:
            if isinstance(validation_data, Dataset):
                val_set = validation_data
            else:
                x_val, y_val = to_tensor(validation_data[0], validation_data[1])
                if not isinstance(x_val, list) and not isinstance(x_val, tuple):
                    x_val = [x_val]
                val_set = TensorDataset(*x_val, y_val)

        del validation_data     # for preventing bugs

        if validation_split is not None:
            val_length = int(len(train_set) * validation_split)
            idx = np.arange(0, len(train_set))
            if shuffle_val_split:
                np.random.seed(1234567890)
                np.random.shuffle(idx)
            train_set, val_set = Subset(train_set, idx[:-val_length]), Subset(train_set, idx[-val_length:])
            if sample_weight is not None:
                sample_weight = [sample_weight[i] for i in idx[:-val_length]]

        if sample_weight is not None:
            assert len(sample_weight) == len(train_set)
            sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
            shuffle = None
        else:
            sampler = None
        
        num_workers = _get_num_workers(num_workers)

        if validation_batch_size is None:
            validation_batch_size = batch_size

        train_loader = DataLoader(train_set, shuffle=shuffle, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
        val_loader = DataLoader(val_set, shuffle=False, batch_size=validation_batch_size, num_workers=num_workers) if has_val else None

        return self.fit_dl(train_loader, val_loader, epochs, callbacks, verbose, use_amp, accum_grad_steps)

    @torch.no_grad()
    def evaluate(self, x, y=None, batch_size=32, num_workers=0, use_amp=False):
        """Return the loss value & metrics values for the model in test mode.\n\n    Computation is done in batches."""
        if isinstance(x, Dataset):
            val_set = x
        else:
            if not isinstance(x, list) and not isinstance(x, tuple):
                x = [x]
            x, y = to_tensor(x, y)
            val_set = TensorDataset(*x, y)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=_get_num_workers(num_workers))
        return self.evaluate_dl(val_loader, use_amp)

    @torch.no_grad()
    def evaluate_dl(self, data_loader, use_amp=False):
        assert self.compiled
        return dict(self.trainer.valid_on_epoch(data_loader, use_amp))

    @torch.no_grad()
    def predict_dl(self, data_loader, device=None, output_numpy=True, activation=None, use_amp=False):
        self._check_keras_layer()
        if device is None:
            if self.compiled:
                device = self.trainer.device
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval().to(device=device)

        activation = _create_activation(activation)

        outputs = []
        for batch in data_loader:
            if not isinstance(batch, list):
                batch = [batch]
                
            for i in range(len(batch)):
                batch[i] = batch[i].to(device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                o = self(*batch)
                outputs.append(o)

        outputs = torch.cat(outputs, dim=0).float()

        if activation is not None:
            outputs = activation(outputs)

        return outputs.cpu().numpy() if output_numpy else outputs

    @torch.no_grad()
    def predict(self, x, batch_size=32, device=None, output_numpy=True, activation=None, num_workers=0, use_amp=False):
        """Generate output predictions for the input samples.\n\n    Computation is done in batches."""
        if isinstance(x, Dataset):
            test_set = x
        else:
            if not isinstance(x, list) and not isinstance(x, tuple):
                x = [x]
            test_set = TensorDataset(*to_tensor(x))
        data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=_get_num_workers(num_workers))
        return self.predict_dl(data_loader, device=device, output_numpy=output_numpy, activation=activation, use_amp=use_amp)

    def save_weights(self, filepath):
        """Equal to `torch.save(model.state_dict(), filepath)`."""
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        """Equal to `model.load_state_dict(torch.load(filepath))`."""
        self.load_state_dict(torch.load(filepath))