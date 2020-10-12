import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torchsummary
from collections import OrderedDict

from ._training import Trainer


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
        return sum([p.numel() for p in self.parameters()])

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
        self.trainer = Trainer(model=self, optimizer=optimizer, loss=loss, metrics=m, device=device)


    def fit(self, x, y, epochs, batch_size=32,
                validation_split=None, shuffle_val_split=False,
                validation_data=None,
                callbacks=[],
                verbose=1,
                precise_mode=False,
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
        history = self.trainer.run(train_loader, val_loader, max_epochs=epochs, verbose=verbose, precise_mode=precise_mode)

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
        self.eval()

        data_loader = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)
        for x_batch in data_loader:
            outputs.append(self.forward(x_batch[0].to(device=device)))

        return torch.cat(outputs, dim=0).cpu().numpy()
