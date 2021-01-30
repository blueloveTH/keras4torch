# Activations

## String Named Activations

```python
OrderedDict({
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(dim=-1),
    'selu': nn.SELU(),
    'celu': nn.CELU(),
    'leaky_relu': nn.LeakyReLU(),
    'relu6': nn.ReLU6(),
    'elu': nn.ELU(),
    'sigmoid': nn.Sigmoid(),
    'mish': Mish(),
    'swish': Swish(),
    'gelu': nn.GELU(),
    'argmax': Lambda(lambda x: x.argmax(-1)),
    'sigmoid_round': Lambda(lambda x: torch.round(torch.sigmoid(x)))
})
```

