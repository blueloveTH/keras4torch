import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from torch.utils.data import random_split
from .._summary import summary

from .._training import Trainer
from ..layers import KerasLayer
from ..metrics import Metric
from ..metrics import _create_metric
from ..losses import _create_loss
from ..optimizers import _create_optimizer
from ..utils import to_tensor

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

        self.has_keras_layer = False
        def check_keras_layer(m):
            if isinstance(m, KerasLayer):
                self.has_keras_layer = True
        self.model.apply(check_keras_layer)

    def forward(self, x):
        return self.model(x)

    def count_params(self) -> int:
        """Count the total number of scalars composing the weights."""
        return sum([p.numel() for p in self.parameters()])


    ########## keras-style methods below ##########

    @torch.no_grad()
    def build(self, input_shape, dtype=torch.float32):
        """Build the model when it contains `KerasLayer`."""
        if self.has_keras_layer:
            input_shape = [2] + list(input_shape)
            probe_input = torch.zeros(size=input_shape).to(dtype=dtype)
            self.model(probe_input)
        self.built = True
        return self

    def _check_keras_layer(self):
        if self.has_keras_layer and not self.built:
            raise AssertionError("You should call `model.build()` first because the model contains `KerasLayer`.")

    def summary(self, input_shape, depth=3):
        """Print a string summary of the network."""
        self._check_keras_layer()
        summary(self.model, input_shape, depth=depth, verbose=1)

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
        if device == None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        loss = _create_loss(loss)
        optimizer = _create_optimizer(optimizer, self.parameters())

        m = OrderedDict({'loss': loss})
        if isinstance(metrics, dict):
            m.update(metrics)
        elif isinstance(metrics, list):
            for tmp_m in metrics:
                tmp_m = _create_metric(tmp_m)
                if isinstance(tmp_m, Metric):
                    m[tmp_m.get_abbr()] = tmp_m
                elif hasattr(tmp_m, '__call__'):
                    m[tmp_m.__name__] = tmp_m
                else:
                    raise TypeError('Unsupported type.')
        elif not (metrics is None):
            raise TypeError('Argument `metrics` should be either a dict or list.')

        self.to(device=device)
        self.trainer = Trainer(model=self, optimizer=optimizer, loss=loss, metrics=m, device=device)
        self.compiled = True


    def fit_dl(self, train_loader, epochs,
                val_loader=None,
                callbacks=[],
                verbose=1,
                precise_train_metrics=False):

        self.trainer.register_callbacks(callbacks)
        history = self.trainer.run(train_loader, val_loader, max_epochs=epochs, verbose=verbose, precise_train_metrics=precise_train_metrics)

        return history


    def fit(self, x, y, epochs, batch_size=32,
                validation_split=None, val_split_seed=7,
                validation_data=None,
                callbacks=[],
                verbose=1,
                precise_train_metrics=False,
                shuffle=True,
                sample_weight=None,
                num_workers=0
                ):
        """Train the model for a fixed number of epochs (iterations on a dataset)."""

        assert self.compiled
        x, y = to_tensor(x, y)

        assert not (validation_data != None and validation_split != None)
        has_val = validation_data != None or validation_split != None

        if type(sample_weight) != type(None):
            if isinstance(sample_weight, list):
                sample_weight = torch.tensor(sample_weight)
            sample_weight = to_tensor(sample_weight).float()
            train_set = TensorDataset(x, y, sample_weight)
        else:
            train_set = TensorDataset(x, y)

        if validation_data != None:
            x_val, y_val = to_tensor(validation_data[0], validation_data[1])
            val_set = TensorDataset(x_val, y_val)

        if validation_split != None:
            val_length = int(len(train_set) * validation_split)
            train_length = len(train_set) - val_length
            train_set, val_set = random_split(train_set, [train_length, val_length], generator=torch.Generator().manual_seed(val_split_seed))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers) if has_val else None

        # Training
        self.trainer.register_callbacks(callbacks)
        history = self.trainer.run(train_loader, val_loader, max_epochs=epochs, verbose=verbose, precise_train_metrics=precise_train_metrics)

        return history

    @torch.no_grad()
    def evaluate(self, x, y, batch_size=32, num_workers=0):
        """Return the loss value & metrics values for the model in test mode.\n\n    Computation is done in batches."""
        x, y = to_tensor(x, y)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        return self.evaluate_dl(val_loader)

    @torch.no_grad()
    def evaluate_dl(self, data_loader):
        assert self.compiled
        return self.trainer.evaluate(data_loader)

    @torch.no_grad()
    def predict_dl(self, data_loader, device=None, output_numpy=True, activation=None):
        self._check_keras_layer()
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

    @torch.no_grad()
    def predict(self, inputs, batch_size=32, device=None, output_numpy=True, activation=None, num_workers=0):
        """
        Generate output predictions for the input samples.\n\n    Computation is done in batches.
        """
        inputs = to_tensor(inputs)
        data_loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        return self.predict_dl(data_loader, device=device, output_numpy=output_numpy, activation=activation)

    def save_weights(self, filepath):
        """Equal to `torch.save(model.state_dict(), filepath)`."""
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        """Equal to `model.load_state_dict(torch.load(filepath))`."""
        self.load_state_dict(torch.load(filepath))
