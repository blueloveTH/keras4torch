import torch
import copy
from ._wrapper import Model

class SymbolicTensor(object):
    def __init__(self, module, *inputs):
        self._module = module
        self._inputs = inputs
    
    @property
    def shape(self):
        return torch.Size([-1] + list(self.eval().shape[1:]))

    def _deep_eval(self, args):
        if isinstance(args, list):
            return [self._deep_eval(i) for i in args]
        else:
            return args.eval()       # leaf

    def eval(self):
        args = [self._deep_eval(i) for i in self._inputs]
        return self._module(*args)

class FunctionalInput(object):
    def __init__(self, input_shape, dtype=torch.float32):
        self._input_shape = torch.Size(input_shape)
        self._dtype = dtype
        self._x = torch.zeros(size=[2] + list(self._input_shape))

    def prepare(self, x):
        self._x = x

    def eval(self):
        return self._x

    @property
    def shape(self):
        return torch.Size([-1] + list(self._input_shape))

    @property
    def dtype(self):
        return self._dtype

class Functional(torch.nn.Module):
    def __init__(self):
        super(Functional, self).__init__()
        self._inputs = None
        self._outputs = None
        self._module_list = []

    def input(self, input_shape, dtype=torch.float32):
        assert self._inputs is None
        self._inputs = FunctionalInput(input_shape=input_shape, dtype=dtype)
        return self._inputs
        
    def call(self, module, *inputs):
        self._module_list.append(module)
        return SymbolicTensor(module, *inputs)

    def build(self, outputs):
        assert self._outputs is None
        self._outputs = outputs
        self._module_list = torch.nn.ModuleList(self._module_list)
        return Model(self).build(self._inputs._input_shape, dtype=self._inputs._dtype)

    def forward(self, x):
        self._inputs.prepare(x)
        return self._outputs.eval()