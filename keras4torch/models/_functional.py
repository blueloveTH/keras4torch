import torch
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

class SymbolicTensorInput(object):
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

class _FunctionalModule(torch.nn.Module):
    def __init__(self):
        super(_FunctionalModule, self).__init__()
        self.inputs = None
        self.outputs = None
    
    def set_modules(self, modules):
        assert self.inputs and self.outputs
        self.module_list = torch.nn.ModuleList(modules)

    @property
    def input_shape(self):
        return self.inputs._input_shape
    
    @property
    def input_dtype(self):
        return self.inputs._dtype

    def forward(self, x):
        self.inputs.prepare(x)
        return self.outputs.eval()

class Functional(object):
    def __init__(self):
        super(Functional, self).__init__()
        self._fn_module = _FunctionalModule()
        self._module_list = []

    def input(self, input_shape, dtype=torch.float32):
        assert self._fn_module.inputs is None
        self._fn_module.inputs = SymbolicTensorInput(input_shape, dtype)
        return self._fn_module.inputs
        
    def __call__(self, module, *inputs):
        self._module_list.append(module)
        return SymbolicTensor(module, *inputs)

    def build_model(self, outputs):
        assert self._fn_module.outputs is None
        self._fn_module.outputs = outputs
        self._fn_module.set_modules(self._module_list)
        return Model(self._fn_module).build(
            self._fn_module.input_shape,
            dtype=self._fn_module.input_dtype
        )