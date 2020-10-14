# Keras4Torch

#### "A Easy To Use Pytorch API for Training PyTorch Models❤"

[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/keras4torch.svg)](https://pypi.python.org/pypi/keras4torch)
[![GitHub license](docs/license-MIT-blue.svg)](https://github.com/blueloveTH/keras4torch)

Keras4Torch is a subset of Keras in PyTorch. You can use `keras4torch.Model` to wrap any `torch.nn.Module` and get the core training features of Keras by using `model.fit()`,  `model.evaluate()` and `model.predict()`. Most of the training code in Keras can work in Keras4Torch with little or no change.If you are a keras user, Keras4Torch would be much perfect for you.

Keras4Torch is a simple tool for training PyTorch model in a keras style. Keras4Torch provide a high-level feature: implementing model training with barely few lines of code.the core code of Keras4Torch are  `keras4torch.Model()`, `model.compile`,`model.fit()`,  `model.evaluate()` and `model.predict()`. If you are a keras enthusiast, Keras4Torch would be much perfect for you.

## Installation

```
pip install keras4torch
```

Keras4Torch supports Python 3.6 or newer.



## Quick Start

Let's start with a simple example of MNIST!

```python
import torch
import torchvision
from torch import nn

import keras4torch
```

#### (1) Preprocess Data

```python
mnist = torchvision.datasets.MNIST(root='./', download=True)
X, y = mnist.train_data, mnist.train_labels

X = X.float() / 255.0    # scale the pixels to [0, 1]

x_train, y_train = X[:40000], y[:40000]
x_test, y_test = X[40000:], y[40000:]
```

#### (2) Define the Model

```python
model = torch.nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512), nn.ReLU(),
    nn.Linear(512, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

model = keras4torch.Model(model)    # attention this line
```

#### (3) Compile Optimizer, Loss and Metric

```python
model.compile(optimizer='adam', loss=nn.CrossEntropyLoss(), metrics=['acc'])
```

#### (4) Training

```python
history = model.fit(x_train, y_train,
                	epochs=30,
                	batch_size=512,
                	validation_split=0.2,
                	)
```

```txt
Train on 32000 samples, validate on 8000 samples:
Epoch 1/30 - 0.7s - loss: 0.7440 - acc: 0.8149 - val_loss: 0.3069 - val_acc: 0.9114 - lr: 1e-03
Epoch 2/30 - 0.5s - loss: 0.2650 - acc: 0.9241 - val_loss: 0.2378 - val_acc: 0.9331 - lr: 1e-03
Epoch 3/30 - 0.5s - loss: 0.1946 - acc: 0.9435 - val_loss: 0.1940 - val_acc: 0.9431 - lr: 1e-03
Epoch 4/30 - 0.5s - loss: 0.1513 - acc: 0.9555 - val_loss: 0.1663 - val_acc: 0.9524 - lr: 1e-03
... ...
```

#### (5) Plot Learning Curve

```
history.plot(kind='line', y=['acc', 'val_acc'])
```

<img src="docs/learning_curve.svg"  />

#### (6) Evaluate on Test Set

```python
model.evaluate(x_test, y_test)
```

```txt
OrderedDict([('loss', 0.121063925), ('acc', 0.9736)])
```



## Feature Support

|                 | keras4torch | torchkeras | keras |
| --------------- | ----------- | ---------- | ----- |
| callbacks       | √           | x          | √     |
| metrics         | √           | √          | √     |
| numpy dataset   | √           | x          | √     |
| GPU support     | √           | √          | √     |
| shape inference | x           | x          | √     |
| functional API  | x           | x          | √     |
| multi-input     | x           | x          | √     |



## Communication

If you have any problems using Keras4Torch, please open a [Github Issues](https://github.com/blueloveTH/keras4torch/issues) or send email to blueloveTH@foxmail.com or zhangzhipengcs@foxmail.com.

We also welcome Pull Requests.

Keras4Torch is still being developing, We are really looking forward to your participation.

Any contribution would be much appreciated : )

