# Layers API

`keras4torch.layers` provides `KerasLayer` for automatic shape inference as well as some other useful layers for quick experiment.

### Built-in KerasLayer

The built-in KerasLayer is a replacement for some native torch module. The followings are supported.

+ Conv1d
+ Conv2d
+ Conv3d
+ Linear
+ GRU
+ LSTM
+ BatchNorm1d
+ BatchNorm2d
+ BatchNorm3d

Compared with native torch modules, what you need to change is omitting the first shape parameter. For example, `nn.Linear(128, 512)` must be rewritten as `k4t.layers.Linear(512)`; `nn.Conv1d(32, 64, kernel_size=3)` must be rewritten as `k4t.layers.Conv1d(64, kernel_size=3)`.



If a model contains `KerasLayer`, you should build it after wrapping it by `keras4torch.Model`.

```python
model = k4t.Model(model) 		# the model contains at least one `KerasLayer`
model.build(input_shape=[128])
```

The argument `input_shape` should not include batch_size dimension.



### Custom KerasLayer

In fact, what a `KerasLayer` do is to delay the module instantiation to the first `module.forward()` call, thus it can get the output shape from its previous layer and decide how many channels should be created.

To write your own `KerasLayer` for automatic shape inference, you need to subclass `KerasLayer` and implement its abstract method `build()`, as the source code shown below.

```python
class KerasLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(KerasLayer, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.module = None

    @abstractclassmethod
    def build(self, in_shape: torch.Size):
        pass

    def forward(self, x):
        if not self.module:
            self.module = self.build(x.shape)
        return self.module.forward(x)
```

In `KerasLayer.build()`, you will get the current input shape to instantiate the actual module and return it. The arguments are stored by `self.args` and `self.kwargs`.

```python
class Conv1d(KerasLayer):
    def build(self, in_shape):
        return nn.Conv1d(in_shape[1], *self.args, **self.kwargs)
```

Note that `KerasLayer` should be used as a wrapper only, which means your native torch module need to be available without `KerasLayer`. Never write logics in `KerasLayer` irrelevant with `in_shape`.



### Others

+ Lambda
+ Add
+ Reshape
+ Concatenate
+ SamePadding