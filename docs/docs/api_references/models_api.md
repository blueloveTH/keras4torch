## Models API

There are two ways to instantiate a `keras4torch.Model`. 

### 1 - Wrap a PyTorch Module

`keras4torch.Model` wraps a `torch.nn.Module` to integrate training and inference features.

#### Training Pipeline

+ compile
+ fit
+ fit_dl
+ evaluate
+ predict

#### Saving & Serialization

+ save_weights
+ load_weights

#### Utilities

+ summary
+ count_params



### 2 - Use Functional API (Beta)

Functional API allows you to build layers by functional programming.

#### Step1: Create a functional object

```python
import keras4torch as k4t

fn = k4t.models.Functional()
```

#### Step2: Define your input

```python
inputs = fn.input([64])
```

`fn.input(input_shape, dtype)` returns a symbolic tensor which has a shape of [batch_size, 64].

#### Step3: Call the layers

```python
seq = fn.call(k4t.layers.Linear(64), inputs)
seq = fn.call(k4t.layers.Add(), [seq, inputs])
output = fn.call(nn.Softmax(-1), seq)
```

This code block generates a residual connection between the original input and the output of the Linear layer, and defines a softmax activation as the final output.

#### Step4: Build the model

```python
model = fn.build(output)
```

This line will build the functional object and convert it to `keras4torch.Model`. You can use it for training at once.