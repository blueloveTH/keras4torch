import torch
from ._wrapper import Model
from ..activations import _create_activation

"""
Bug warning:
fn = k4t.models.Functional()

>>> inputs = fn.input([28, 28])
>>> seq = fn(k4t.layers.SqueezeAndExcitation1d(), inputs)
>>> seq = fn(nn.Flatten(), inputs)          # leakage!
>>> seq = fn(k4t.layers.Linear(10), seq)
>>> model = fn.build_model(seq)

>>> model.summary()     # will cause an strange error due to the leakage of the first `seq`.
"""

class SymbolicTensorBase(object):
    def eval(self):
        raise NotImplementedError

    def get_link_info(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_link_info()

    @property
    def shape(self) -> torch.Size:
        return torch.Size([-1] + list(self.eval().shape[1:]))
    

class SymbolicTensor(SymbolicTensorBase):
    def __init__(self, module, *inputs, **kwargs):
        self._module = module
        self._inputs = inputs
        self._cached_output = None

    def _clear_cache(self):
        self._cached_output = None

    def _deep_eval(self, args):
        if isinstance(args, list) or isinstance(args, tuple):
            return type(args)([self._deep_eval(i) for i in args])
        else:
            return args.eval()       # leaf

    def eval(self):
        if self._cached_output is None:
            args = [self._deep_eval(i) for i in self._inputs]
            self._cached_output = self._module(*args)
        return self._cached_output


    def get_link_info(self) -> str:
        """tmp code"""
        inputs_str = ', '.join([
            str(i) if not isinstance(i, list) else str([str(j) for j in i]) for i in self._inputs 
            ])
        module_name = str(self._module.__class__).split('.')[-1][:-2]
        return f"({inputs_str}) -> {module_name}"

class SymbolicTensorInput(SymbolicTensorBase):
    def __init__(self, input_shape, dtype=torch.float32):
        self._input_shape = torch.Size(input_shape)
        self._dtype = dtype
        self._x = torch.zeros(size=[2] + list(self._input_shape))

    def prepare(self, x):
        self._x = x

    def eval(self):
        return self._x

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def dtype(self):
        return self._dtype

    def get_link_info(self):
        """tmp code"""
        return 'Input'
        #return f'Input({list(self.shape)})'

class _FunctionalModule(torch.nn.ModuleList):
    def __init__(self):
        super(_FunctionalModule, self).__init__()
        self.inputs = None
        self.outputs = None
    
    def initialize(self, modules, symbolic_tensors):
        assert self.inputs and self.outputs
        self.extend(modules)
        self.symbolic_tensors = symbolic_tensors

    @property
    def input_shape(self):
        return self.inputs.input_shape
    
    @property
    def input_dtype(self):
        return self.inputs._dtype

    def forward(self, x):
        for st in self.symbolic_tensors:
            st._clear_cache()
        self.inputs.prepare(x)
        return self.outputs.eval()

class Functional(object):
    def __init__(self):
        super(Functional, self).__init__()
        self._fn_module = _FunctionalModule()
        self._modules = set()
        self._symbolic_tensors = []

        print('\033[33m' + '[Warning] Functional API is a beta feature. Do not use it for production.')

    def input(self, input_shape, dtype=torch.float32):
        assert self._fn_module.inputs is None
        self._fn_module.inputs = SymbolicTensorInput(input_shape, dtype)
        return self._fn_module.inputs
        
    def __call__(self, module, *inputs, activation=None, **kwargs):
        self._modules.add(module)
        st = SymbolicTensor(module, *inputs, **kwargs)
        self._symbolic_tensors.append(st)
        
        if activation is not None:
            activation = _create_activation(activation)
            st = self(activation, st)

        return st

    def build_model(self, outputs):
        assert self._fn_module.outputs is None
        self._fn_module.outputs = outputs
        self._fn_module.initialize(self._modules, self._symbolic_tensors)

        return Model(self._fn_module).build(
            self._fn_module.input_shape,
            dtype=self._fn_module.input_dtype
        )