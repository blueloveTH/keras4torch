import torch
from ._wrapper import Model

class _FunctionalLayer(object):
    def __init__(self, module, inputs):
        #super(_FunctionalLayer, self).__init__()
        self.module = module
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.inputs = inputs
    
    @property
    def shape(self):
        return self.forward().shape

    def forward(self):
        args = [i.forward() for i in self.inputs]
        return self.module(*args)

class _FunctionalInput(object):
    def __init__(self, input_shape, dtype=torch.float32):
        #super(_FunctionalInput, self).__init__()
        self._input_shape = torch.Size(input_shape)
        self._dtype = dtype
        self._x = torch.zeros(size=[2] + list(self._input_shape))

    def prepare(self, x):
        self._x = x

    def forward(self):
        return self._x

    @property
    def shape(self):
        return self._input_shape
    @property
    def dtype(self):
        return self._dtype

class Functional(torch.nn.Module):
    def __init__(self):
        super(Functional, self).__init__()
        self.input_layer = None
        self.output_layer = None
        self.module_list = []

    def input(self, input_shape, dtype=torch.float32):
        assert self.input_layer is None
        self.input_layer = _FunctionalInput(input_shape=input_shape, dtype=dtype)
        return self.input_layer
        
    def call(self, module, inputs):
        layer = _FunctionalLayer(module, inputs)
        self.module_list.append(module)
        return layer

    def build(self, output_layer):
        assert self.output_layer is None
        self.output_layer = output_layer
        self.module_list = torch.nn.ModuleList(self.module_list)
        model = Model(self)
        model.build(self.input_layer.shape, dtype=self.input_layer.dtype)
        return model

    def forward(self, x):
        self.input_layer.prepare(x)
        return self.output_layer.forward()