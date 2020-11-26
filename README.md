# Keras4Torch

####  **[README in English](https://github.com/blueloveTH/keras4torch/blob/main/README_en.md)**

####  “开箱即用”的PyTorch模型训练高级API

+ 对keras爱好者来说，Keras4Torch保留了绝大多数的Keras特性。你能够使用和keras相同的代码运行pytorch模型。

+ 对pytorch爱好者来说，Keras4Torch使你只需要几行代码就可以完成pytorch模型的训练、评估和推理。

[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org)
[![PyTorch Versions](https://img.shields.io/badge/PyTorch-1.6+-blue.svg)](https://pypi.org/project/keras4torch)
[![Downloads](https://pepy.tech/badge/keras4torch)](https://pepy.tech/project/keras4torch)
[![pypi](https://img.shields.io/pypi/v/keras4torch.svg)](https://pypi.python.org/pypi/keras4torch)
[![License](https://img.shields.io/github/license/blueloveTH/keras4torch.svg)](https://github.com/blueloveTH/keras4torch/blob/master/LICENSE)



## 安装与配置

```
pip install keras4torch
```

支持PyTorch 1.6及以上。使用早期版本的torch可能导致部分功能不可用。



## 快速开始

作为示例，让我们开始编写一个MNIST手写数字识别程序！

```python
import torch
import torchvision
from torch import nn

import keras4torch
```

#### Step1: 数据预处理

首先，从`torchvision.datasets`中加载MNIST数据集，并将每个像素点缩放到[0, 1]之间。

其中前40000张图片作为训练集，后20000张图片作为测试集。

```python
mnist = torchvision.datasets.MNIST(root='./', download=True)
X, y = mnist.train_data, mnist.train_labels

X = X.float() / 255.0

x_train, y_train = X[:40000], y[:40000]
x_test, y_test = X[40000:], y[40000:]
```

#### Step2: 构建模型

我们使用`torch.nn.Sequential`定义一个由三层全连接组成的线性模型，激活函数为ReLU。

接着，使用`keras4torch.Model`封装Sequential模型，以集成训练API。

```python
model = torch.nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512), nn.ReLU(),
    nn.Linear(512, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

model = keras4torch.Model(model)
```

**News (v0.4.1):** 您也可以使用keras4torch.layers提供的`KerasLayer`，以自动推算输入维度。

包含`KerasLayer`的模型需要调用`model.build()`，其参数是样本的维度。具体示例如下：

```python
import keras4torch.layers as layers

model = torch.nn.Sequential(
    nn.Flatten(),
    layers.Linear(512), nn.ReLU(),
    layers.Linear(128), nn.ReLU(),
    layers.Linear(10)
)

model = keras4torch.Model(model).build(input_shape=[28, 28])
```

#### Step3: 设置优化器、损失函数和度量

`model.compile()`函数对模型进行必要的配置。

参数既可以使用字符串，也可以使用`torch.nn`模块中提供的类实例。

```python
model.compile(optimizer='adam', loss=nn.CrossEntropyLoss(), metrics=['acc'])
```

#### Step4: 训练模型

`model.fit()`是训练模型的方法，将以batch_size=512运行30轮次。

`validation_split=0.2`指定80%数据用于训练集，剩余20%用作验证集。

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

#### Step5: 打印学习曲线

`model.fit()`方法在结束时，返回关于训练历史数据的`pandas.DataFrame`实例。

```
history.plot(kind='line', y=['acc', 'val_acc'])
```

<img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiIHN0YW5kYWxvbmU9Im5vIj8+DQo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDEuMS8vRU4iDQogICJodHRwOi8vd3d3LnczLm9yZy9HcmFwaGljcy9TVkcvMS4xL0RURC9zdmcxMS5kdGQiPg0KPCEtLSBDcmVhdGVkIHdpdGggbWF0cGxvdGxpYiAoaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pIC0tPg0KPHN2ZyBoZWlnaHQ9IjI0OC41MTgxMjVwdCIgdmVyc2lvbj0iMS4xIiB2aWV3Qm94PSIwIDAgMzg0LjgyODEyNSAyNDguNTE4MTI1IiB3aWR0aD0iMzg0LjgyODEyNXB0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj4NCiA8bWV0YWRhdGE+DQogIDxyZGY6UkRGIHhtbG5zOmNjPSJodHRwOi8vY3JlYXRpdmVjb21tb25zLm9yZy9ucyMiIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4NCiAgIDxjYzpXb3JrPg0KICAgIDxkYzp0eXBlIHJkZjpyZXNvdXJjZT0iaHR0cDovL3B1cmwub3JnL2RjL2RjbWl0eXBlL1N0aWxsSW1hZ2UiLz4NCiAgICA8ZGM6ZGF0ZT4yMDIwLTEwLTEzVDIyOjUzOjI2Ljc0MTM3NzwvZGM6ZGF0ZT4NCiAgICA8ZGM6Zm9ybWF0PmltYWdlL3N2Zyt4bWw8L2RjOmZvcm1hdD4NCiAgICA8ZGM6Y3JlYXRvcj4NCiAgICAgPGNjOkFnZW50Pg0KICAgICAgPGRjOnRpdGxlPk1hdHBsb3RsaWIgdjMuMy4yLCBodHRwczovL21hdHBsb3RsaWIub3JnLzwvZGM6dGl0bGU+DQogICAgIDwvY2M6QWdlbnQ+DQogICAgPC9kYzpjcmVhdG9yPg0KICAgPC9jYzpXb3JrPg0KICA8L3JkZjpSREY+DQogPC9tZXRhZGF0YT4NCiA8ZGVmcz4NCiAgPHN0eWxlIHR5cGU9InRleHQvY3NzIj4qe3N0cm9rZS1saW5lY2FwOmJ1dHQ7c3Ryb2tlLWxpbmVqb2luOnJvdW5kO308L3N0eWxlPg0KIDwvZGVmcz4NCiA8ZyBpZD0iZmlndXJlXzEiPg0KICA8ZyBpZD0icGF0Y2hfMSI+DQogICA8cGF0aCBkPSJNIDAgMjQ4LjUxODEyNSANCkwgMzg0LjgyODEyNSAyNDguNTE4MTI1IA0KTCAzODQuODI4MTI1IDAgDQpMIDAgMCANCnoNCiIgc3R5bGU9ImZpbGw6bm9uZTsiLz4NCiAgPC9nPg0KICA8ZyBpZD0iYXhlc18xIj4NCiAgIDxnIGlkPSJwYXRjaF8yIj4NCiAgICA8cGF0aCBkPSJNIDQyLjgyODEyNSAyMjQuNjQgDQpMIDM3Ny42MjgxMjUgMjI0LjY0IA0KTCAzNzcuNjI4MTI1IDcuMiANCkwgNDIuODI4MTI1IDcuMiANCnoNCiIgc3R5bGU9ImZpbGw6I2ZmZmZmZjsiLz4NCiAgIDwvZz4NCiAgIDxnIGlkPSJtYXRwbG90bGliLmF4aXNfMSI+DQogICAgPGcgaWQ9Inh0aWNrXzEiPg0KICAgICA8ZyBpZD0ibGluZTJkXzEiPg0KICAgICAgPGRlZnM+DQogICAgICAgPHBhdGggZD0iTSAwIDAgDQpMIDAgMy41IA0KIiBpZD0ibTk2YjVjNzExODEiIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIvPg0KICAgICAgPC9kZWZzPg0KICAgICAgPGc+DQogICAgICAgPHVzZSBzdHlsZT0ic3Ryb2tlOiMwMDAwMDA7c3Ryb2tlLXdpZHRoOjAuODsiIHg9IjQ3LjU1MTAwOSIgeGxpbms6aHJlZj0iI205NmI1YzcxMTgxIiB5PSIyMjQuNjQiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgICA8ZyBpZD0idGV4dF8xIj4NCiAgICAgIDwhLS0gMCAtLT4NCiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ0LjM2OTc1OSAyMzkuMjM4NDM3KXNjYWxlKDAuMSAtMC4xKSI+DQogICAgICAgPGRlZnM+DQogICAgICAgIDxwYXRoIGQ9Ik0gMzEuNzgxMjUgNjYuNDA2MjUgDQpRIDI0LjE3MTg3NSA2Ni40MDYyNSAyMC4zMjgxMjUgNTguOTA2MjUgDQpRIDE2LjUgNTEuNDIxODc1IDE2LjUgMzYuMzc1IA0KUSAxNi41IDIxLjM5MDYyNSAyMC4zMjgxMjUgMTMuODkwNjI1IA0KUSAyNC4xNzE4NzUgNi4zOTA2MjUgMzEuNzgxMjUgNi4zOTA2MjUgDQpRIDM5LjQ1MzEyNSA2LjM5MDYyNSA0My4yODEyNSAxMy44OTA2MjUgDQpRIDQ3LjEyNSAyMS4zOTA2MjUgNDcuMTI1IDM2LjM3NSANClEgNDcuMTI1IDUxLjQyMTg3NSA0My4yODEyNSA1OC45MDYyNSANClEgMzkuNDUzMTI1IDY2LjQwNjI1IDMxLjc4MTI1IDY2LjQwNjI1IA0Keg0KTSAzMS43ODEyNSA3NC4yMTg3NSANClEgNDQuMDQ2ODc1IDc0LjIxODc1IDUwLjUxNTYyNSA2NC41MTU2MjUgDQpRIDU2Ljk4NDM3NSA1NC44MjgxMjUgNTYuOTg0Mzc1IDM2LjM3NSANClEgNTYuOTg0Mzc1IDE3Ljk2ODc1IDUwLjUxNTYyNSA4LjI2NTYyNSANClEgNDQuMDQ2ODc1IC0xLjQyMTg3NSAzMS43ODEyNSAtMS40MjE4NzUgDQpRIDE5LjUzMTI1IC0xLjQyMTg3NSAxMy4wNjI1IDguMjY1NjI1IA0KUSA2LjU5Mzc1IDE3Ljk2ODc1IDYuNTkzNzUgMzYuMzc1IA0KUSA2LjU5Mzc1IDU0LjgyODEyNSAxMy4wNjI1IDY0LjUxNTYyNSANClEgMTkuNTMxMjUgNzQuMjE4NzUgMzEuNzgxMjUgNzQuMjE4NzUgDQp6DQoiIGlkPSJEZWphVnVTYW5zLTQ4Ii8+DQogICAgICAgPC9kZWZzPg0KICAgICAgIDx1c2UgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDgiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ieHRpY2tfMiI+DQogICAgIDxnIGlkPSJsaW5lMmRfMiI+DQogICAgICA8Zz4NCiAgICAgICA8dXNlIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIgeD0iMTAwLjAyNzQ5OCIgeGxpbms6aHJlZj0iI205NmI1YzcxMTgxIiB5PSIyMjQuNjQiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgICA8ZyBpZD0idGV4dF8yIj4NCiAgICAgIDwhLS0gNSAtLT4NCiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDk2Ljg0NjI0OCAyMzkuMjM4NDM3KXNjYWxlKDAuMSAtMC4xKSI+DQogICAgICAgPGRlZnM+DQogICAgICAgIDxwYXRoIGQ9Ik0gMTAuNzk2ODc1IDcyLjkwNjI1IA0KTCA0OS41MTU2MjUgNzIuOTA2MjUgDQpMIDQ5LjUxNTYyNSA2NC41OTM3NSANCkwgMTkuODI4MTI1IDY0LjU5Mzc1IA0KTCAxOS44MjgxMjUgNDYuNzM0Mzc1IA0KUSAyMS45Njg3NSA0Ny40Njg3NSAyNC4xMDkzNzUgNDcuODI4MTI1IA0KUSAyNi4yNjU2MjUgNDguMTg3NSAyOC40MjE4NzUgNDguMTg3NSANClEgNDAuNjI1IDQ4LjE4NzUgNDcuNzUgNDEuNSANClEgNTQuODkwNjI1IDM0LjgxMjUgNTQuODkwNjI1IDIzLjM5MDYyNSANClEgNTQuODkwNjI1IDExLjYyNSA0Ny41NjI1IDUuMDkzNzUgDQpRIDQwLjIzNDM3NSAtMS40MjE4NzUgMjYuOTA2MjUgLTEuNDIxODc1IA0KUSAyMi4zMTI1IC0xLjQyMTg3NSAxNy41NDY4NzUgLTAuNjQwNjI1IA0KUSAxMi43OTY4NzUgMC4xNDA2MjUgNy43MTg3NSAxLjcwMzEyNSANCkwgNy43MTg3NSAxMS42MjUgDQpRIDEyLjEwOTM3NSA5LjIzNDM3NSAxNi43OTY4NzUgOC4wNjI1IA0KUSAyMS40ODQzNzUgNi44OTA2MjUgMjYuNzAzMTI1IDYuODkwNjI1IA0KUSAzNS4xNTYyNSA2Ljg5MDYyNSA0MC4wNzgxMjUgMTEuMzI4MTI1IA0KUSA0NS4wMTU2MjUgMTUuNzY1NjI1IDQ1LjAxNTYyNSAyMy4zOTA2MjUgDQpRIDQ1LjAxNTYyNSAzMSA0MC4wNzgxMjUgMzUuNDM3NSANClEgMzUuMTU2MjUgMzkuODkwNjI1IDI2LjcwMzEyNSAzOS44OTA2MjUgDQpRIDIyLjc1IDM5Ljg5MDYyNSAxOC44MTI1IDM5LjAxNTYyNSANClEgMTQuODkwNjI1IDM4LjE0MDYyNSAxMC43OTY4NzUgMzYuMjgxMjUgDQp6DQoiIGlkPSJEZWphVnVTYW5zLTUzIi8+DQogICAgICAgPC9kZWZzPg0KICAgICAgIDx1c2UgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTMiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ieHRpY2tfMyI+DQogICAgIDxnIGlkPSJsaW5lMmRfMyI+DQogICAgICA8Zz4NCiAgICAgICA8dXNlIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIgeD0iMTUyLjUwMzk4NyIgeGxpbms6aHJlZj0iI205NmI1YzcxMTgxIiB5PSIyMjQuNjQiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgICA8ZyBpZD0idGV4dF8zIj4NCiAgICAgIDwhLS0gMTAgLS0+DQogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNDYuMTQxNDg3IDIzOS4yMzg0Mzcpc2NhbGUoMC4xIC0wLjEpIj4NCiAgICAgICA8ZGVmcz4NCiAgICAgICAgPHBhdGggZD0iTSAxMi40MDYyNSA4LjI5Njg3NSANCkwgMjguNTE1NjI1IDguMjk2ODc1IA0KTCAyOC41MTU2MjUgNjMuOTIxODc1IA0KTCAxMC45ODQzNzUgNjAuNDA2MjUgDQpMIDEwLjk4NDM3NSA2OS4zOTA2MjUgDQpMIDI4LjQyMTg3NSA3Mi45MDYyNSANCkwgMzguMjgxMjUgNzIuOTA2MjUgDQpMIDM4LjI4MTI1IDguMjk2ODc1IA0KTCA1NC4zOTA2MjUgOC4yOTY4NzUgDQpMIDU0LjM5MDYyNSAwIA0KTCAxMi40MDYyNSAwIA0Keg0KIiBpZD0iRGVqYVZ1U2Fucy00OSIvPg0KICAgICAgIDwvZGVmcz4NCiAgICAgICA8dXNlIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ5Ii8+DQogICAgICAgPHVzZSB4PSI2My42MjMwNDciIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ4Ii8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICA8L2c+DQogICAgPGcgaWQ9Inh0aWNrXzQiPg0KICAgICA8ZyBpZD0ibGluZTJkXzQiPg0KICAgICAgPGc+DQogICAgICAgPHVzZSBzdHlsZT0ic3Ryb2tlOiMwMDAwMDA7c3Ryb2tlLXdpZHRoOjAuODsiIHg9IjIwNC45ODA0NzYiIHhsaW5rOmhyZWY9IiNtOTZiNWM3MTE4MSIgeT0iMjI0LjY0Ii8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICAgPGcgaWQ9InRleHRfNCI+DQogICAgICA8IS0tIDE1IC0tPg0KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTk4LjYxNzk3NiAyMzkuMjM4NDM3KXNjYWxlKDAuMSAtMC4xKSI+DQogICAgICAgPHVzZSB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OSIvPg0KICAgICAgIDx1c2UgeD0iNjMuNjIzMDQ3IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy01MyIvPg0KICAgICAgPC9nPg0KICAgICA8L2c+DQogICAgPC9nPg0KICAgIDxnIGlkPSJ4dGlja181Ij4NCiAgICAgPGcgaWQ9ImxpbmUyZF81Ij4NCiAgICAgIDxnPg0KICAgICAgIDx1c2Ugc3R5bGU9InN0cm9rZTojMDAwMDAwO3N0cm9rZS13aWR0aDowLjg7IiB4PSIyNTcuNDU2OTY1IiB4bGluazpocmVmPSIjbTk2YjVjNzExODEiIHk9IjIyNC42NCIvPg0KICAgICAgPC9nPg0KICAgICA8L2c+DQogICAgIDxnIGlkPSJ0ZXh0XzUiPg0KICAgICAgPCEtLSAyMCAtLT4NCiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDI1MS4wOTQ0NjUgMjM5LjIzODQzNylzY2FsZSgwLjEgLTAuMSkiPg0KICAgICAgIDxkZWZzPg0KICAgICAgICA8cGF0aCBkPSJNIDE5LjE4NzUgOC4yOTY4NzUgDQpMIDUzLjYwOTM3NSA4LjI5Njg3NSANCkwgNTMuNjA5Mzc1IDAgDQpMIDcuMzI4MTI1IDAgDQpMIDcuMzI4MTI1IDguMjk2ODc1IA0KUSAxMi45Mzc1IDE0LjEwOTM3NSAyMi42MjUgMjMuODkwNjI1IA0KUSAzMi4zMjgxMjUgMzMuNjg3NSAzNC44MTI1IDM2LjUzMTI1IA0KUSAzOS41NDY4NzUgNDEuODQzNzUgNDEuNDIxODc1IDQ1LjUzMTI1IA0KUSA0My4zMTI1IDQ5LjIxODc1IDQzLjMxMjUgNTIuNzgxMjUgDQpRIDQzLjMxMjUgNTguNTkzNzUgMzkuMjM0Mzc1IDYyLjI1IA0KUSAzNS4xNTYyNSA2NS45MjE4NzUgMjguNjA5Mzc1IDY1LjkyMTg3NSANClEgMjMuOTY4NzUgNjUuOTIxODc1IDE4LjgxMjUgNjQuMzEyNSANClEgMTMuNjcxODc1IDYyLjcwMzEyNSA3LjgxMjUgNTkuNDIxODc1IA0KTCA3LjgxMjUgNjkuMzkwNjI1IA0KUSAxMy43NjU2MjUgNzEuNzgxMjUgMTguOTM3NSA3MyANClEgMjQuMTI1IDc0LjIxODc1IDI4LjQyMTg3NSA3NC4yMTg3NSANClEgMzkuNzUgNzQuMjE4NzUgNDYuNDg0Mzc1IDY4LjU0Njg3NSANClEgNTMuMjE4NzUgNjIuODkwNjI1IDUzLjIxODc1IDUzLjQyMTg3NSANClEgNTMuMjE4NzUgNDguOTIxODc1IDUxLjUzMTI1IDQ0Ljg5MDYyNSANClEgNDkuODU5Mzc1IDQwLjg3NSA0NS40MDYyNSAzNS40MDYyNSANClEgNDQuMTg3NSAzMy45ODQzNzUgMzcuNjQwNjI1IDI3LjIxODc1IA0KUSAzMS4xMDkzNzUgMjAuNDUzMTI1IDE5LjE4NzUgOC4yOTY4NzUgDQp6DQoiIGlkPSJEZWphVnVTYW5zLTUwIi8+DQogICAgICAgPC9kZWZzPg0KICAgICAgIDx1c2UgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTAiLz4NCiAgICAgICA8dXNlIHg9IjYzLjYyMzA0NyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDgiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ieHRpY2tfNiI+DQogICAgIDxnIGlkPSJsaW5lMmRfNiI+DQogICAgICA8Zz4NCiAgICAgICA8dXNlIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIgeD0iMzA5LjkzMzQ1NCIgeGxpbms6aHJlZj0iI205NmI1YzcxMTgxIiB5PSIyMjQuNjQiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgICA8ZyBpZD0idGV4dF82Ij4NCiAgICAgIDwhLS0gMjUgLS0+DQogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzMDMuNTcwOTU0IDIzOS4yMzg0Mzcpc2NhbGUoMC4xIC0wLjEpIj4NCiAgICAgICA8dXNlIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTUwIi8+DQogICAgICAgPHVzZSB4PSI2My42MjMwNDciIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTUzIi8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICA8L2c+DQogICAgPGcgaWQ9Inh0aWNrXzciPg0KICAgICA8ZyBpZD0ibGluZTJkXzciPg0KICAgICAgPGc+DQogICAgICAgPHVzZSBzdHlsZT0ic3Ryb2tlOiMwMDAwMDA7c3Ryb2tlLXdpZHRoOjAuODsiIHg9IjM2Mi40MDk5NDMiIHhsaW5rOmhyZWY9IiNtOTZiNWM3MTE4MSIgeT0iMjI0LjY0Ii8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICAgPGcgaWQ9InRleHRfNyI+DQogICAgICA8IS0tIDMwIC0tPg0KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzU2LjA0NzQ0MyAyMzkuMjM4NDM3KXNjYWxlKDAuMSAtMC4xKSI+DQogICAgICAgPGRlZnM+DQogICAgICAgIDxwYXRoIGQ9Ik0gNDAuNTc4MTI1IDM5LjMxMjUgDQpRIDQ3LjY1NjI1IDM3Ljc5Njg3NSA1MS42MjUgMzMgDQpRIDU1LjYwOTM3NSAyOC4yMTg3NSA1NS42MDkzNzUgMjEuMTg3NSANClEgNTUuNjA5Mzc1IDEwLjQwNjI1IDQ4LjE4NzUgNC40ODQzNzUgDQpRIDQwLjc2NTYyNSAtMS40MjE4NzUgMjcuMDkzNzUgLTEuNDIxODc1IA0KUSAyMi41MTU2MjUgLTEuNDIxODc1IDE3LjY1NjI1IC0wLjUxNTYyNSANClEgMTIuNzk2ODc1IDAuMzkwNjI1IDcuNjI1IDIuMjAzMTI1IA0KTCA3LjYyNSAxMS43MTg3NSANClEgMTEuNzE4NzUgOS4zMjgxMjUgMTYuNTkzNzUgOC4xMDkzNzUgDQpRIDIxLjQ4NDM3NSA2Ljg5MDYyNSAyNi44MTI1IDYuODkwNjI1IA0KUSAzNi4wNzgxMjUgNi44OTA2MjUgNDAuOTM3NSAxMC41NDY4NzUgDQpRIDQ1Ljc5Njg3NSAxNC4yMDMxMjUgNDUuNzk2ODc1IDIxLjE4NzUgDQpRIDQ1Ljc5Njg3NSAyNy42NDA2MjUgNDEuMjgxMjUgMzEuMjY1NjI1IA0KUSAzNi43NjU2MjUgMzQuOTA2MjUgMjguNzE4NzUgMzQuOTA2MjUgDQpMIDIwLjIxODc1IDM0LjkwNjI1IA0KTCAyMC4yMTg3NSA0My4wMTU2MjUgDQpMIDI5LjEwOTM3NSA0My4wMTU2MjUgDQpRIDM2LjM3NSA0My4wMTU2MjUgNDAuMjM0Mzc1IDQ1LjkyMTg3NSANClEgNDQuMDkzNzUgNDguODI4MTI1IDQ0LjA5Mzc1IDU0LjI5Njg3NSANClEgNDQuMDkzNzUgNTkuOTA2MjUgNDAuMTA5Mzc1IDYyLjkwNjI1IA0KUSAzNi4xNDA2MjUgNjUuOTIxODc1IDI4LjcxODc1IDY1LjkyMTg3NSANClEgMjQuNjU2MjUgNjUuOTIxODc1IDIwLjAxNTYyNSA2NS4wMzEyNSANClEgMTUuMzc1IDY0LjE1NjI1IDkuODEyNSA2Mi4zMTI1IA0KTCA5LjgxMjUgNzEuMDkzNzUgDQpRIDE1LjQzNzUgNzIuNjU2MjUgMjAuMzQzNzUgNzMuNDM3NSANClEgMjUuMjUgNzQuMjE4NzUgMjkuNTkzNzUgNzQuMjE4NzUgDQpRIDQwLjgyODEyNSA3NC4yMTg3NSA0Ny4zNTkzNzUgNjkuMTA5Mzc1IA0KUSA1My45MDYyNSA2NC4wMTU2MjUgNTMuOTA2MjUgNTUuMzI4MTI1IA0KUSA1My45MDYyNSA0OS4yNjU2MjUgNTAuNDM3NSA0NS4wOTM3NSANClEgNDYuOTY4NzUgNDAuOTIxODc1IDQwLjU3ODEyNSAzOS4zMTI1IA0Keg0KIiBpZD0iRGVqYVZ1U2Fucy01MSIvPg0KICAgICAgIDwvZGVmcz4NCiAgICAgICA8dXNlIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTUxIi8+DQogICAgICAgPHVzZSB4PSI2My42MjMwNDciIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ4Ii8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICA8L2c+DQogICA8L2c+DQogICA8ZyBpZD0ibWF0cGxvdGxpYi5heGlzXzIiPg0KICAgIDxnIGlkPSJ5dGlja18xIj4NCiAgICAgPGcgaWQ9ImxpbmUyZF84Ij4NCiAgICAgIDxkZWZzPg0KICAgICAgIDxwYXRoIGQ9Ik0gMCAwIA0KTCAtMy41IDAgDQoiIGlkPSJtYThmYzFkNTA1ZiIgc3R5bGU9InN0cm9rZTojMDAwMDAwO3N0cm9rZS13aWR0aDowLjg7Ii8+DQogICAgICA8L2RlZnM+DQogICAgICA8Zz4NCiAgICAgICA8dXNlIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIgeD0iNDIuODI4MTI1IiB4bGluazpocmVmPSIjbWE4ZmMxZDUwNWYiIHk9IjIwNC4wMTk5NzQiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgICA8ZyBpZD0idGV4dF84Ij4NCiAgICAgIDwhLS0gMC44MjUgLS0+DQogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3LjIgMjA3LjgxOTE5MylzY2FsZSgwLjEgLTAuMSkiPg0KICAgICAgIDxkZWZzPg0KICAgICAgICA8cGF0aCBkPSJNIDEwLjY4NzUgMTIuNDA2MjUgDQpMIDIxIDEyLjQwNjI1IA0KTCAyMSAwIA0KTCAxMC42ODc1IDAgDQp6DQoiIGlkPSJEZWphVnVTYW5zLTQ2Ii8+DQogICAgICAgIDxwYXRoIGQ9Ik0gMzEuNzgxMjUgMzQuNjI1IA0KUSAyNC43NSAzNC42MjUgMjAuNzE4NzUgMzAuODU5Mzc1IA0KUSAxNi43MDMxMjUgMjcuMDkzNzUgMTYuNzAzMTI1IDIwLjUxNTYyNSANClEgMTYuNzAzMTI1IDEzLjkyMTg3NSAyMC43MTg3NSAxMC4xNTYyNSANClEgMjQuNzUgNi4zOTA2MjUgMzEuNzgxMjUgNi4zOTA2MjUgDQpRIDM4LjgxMjUgNi4zOTA2MjUgNDIuODU5Mzc1IDEwLjE3MTg3NSANClEgNDYuOTIxODc1IDEzLjk2ODc1IDQ2LjkyMTg3NSAyMC41MTU2MjUgDQpRIDQ2LjkyMTg3NSAyNy4wOTM3NSA0Mi44OTA2MjUgMzAuODU5Mzc1IA0KUSAzOC44NzUgMzQuNjI1IDMxLjc4MTI1IDM0LjYyNSANCnoNCk0gMjEuOTIxODc1IDM4LjgxMjUgDQpRIDE1LjU3ODEyNSA0MC4zNzUgMTIuMDMxMjUgNDQuNzE4NzUgDQpRIDguNSA0OS4wNzgxMjUgOC41IDU1LjMyODEyNSANClEgOC41IDY0LjA2MjUgMTQuNzE4NzUgNjkuMTQwNjI1IA0KUSAyMC45NTMxMjUgNzQuMjE4NzUgMzEuNzgxMjUgNzQuMjE4NzUgDQpRIDQyLjY3MTg3NSA3NC4yMTg3NSA0OC44NzUgNjkuMTQwNjI1IA0KUSA1NS4wNzgxMjUgNjQuMDYyNSA1NS4wNzgxMjUgNTUuMzI4MTI1IA0KUSA1NS4wNzgxMjUgNDkuMDc4MTI1IDUxLjUzMTI1IDQ0LjcxODc1IA0KUSA0OCA0MC4zNzUgNDEuNzAzMTI1IDM4LjgxMjUgDQpRIDQ4LjgyODEyNSAzNy4xNTYyNSA1Mi43OTY4NzUgMzIuMzEyNSANClEgNTYuNzgxMjUgMjcuNDg0Mzc1IDU2Ljc4MTI1IDIwLjUxNTYyNSANClEgNTYuNzgxMjUgOS45MDYyNSA1MC4zMTI1IDQuMjM0Mzc1IA0KUSA0My44NDM3NSAtMS40MjE4NzUgMzEuNzgxMjUgLTEuNDIxODc1IA0KUSAxOS43MzQzNzUgLTEuNDIxODc1IDEzLjI1IDQuMjM0Mzc1IA0KUSA2Ljc4MTI1IDkuOTA2MjUgNi43ODEyNSAyMC41MTU2MjUgDQpRIDYuNzgxMjUgMjcuNDg0Mzc1IDEwLjc4MTI1IDMyLjMxMjUgDQpRIDE0Ljc5Njg3NSAzNy4xNTYyNSAyMS45MjE4NzUgMzguODEyNSANCnoNCk0gMTguMzEyNSA1NC4zOTA2MjUgDQpRIDE4LjMxMjUgNDguNzM0Mzc1IDIxLjg0Mzc1IDQ1LjU2MjUgDQpRIDI1LjM5MDYyNSA0Mi4zOTA2MjUgMzEuNzgxMjUgNDIuMzkwNjI1IA0KUSAzOC4xNDA2MjUgNDIuMzkwNjI1IDQxLjcxODc1IDQ1LjU2MjUgDQpRIDQ1LjMxMjUgNDguNzM0Mzc1IDQ1LjMxMjUgNTQuMzkwNjI1IA0KUSA0NS4zMTI1IDYwLjA2MjUgNDEuNzE4NzUgNjMuMjM0Mzc1IA0KUSAzOC4xNDA2MjUgNjYuNDA2MjUgMzEuNzgxMjUgNjYuNDA2MjUgDQpRIDI1LjM5MDYyNSA2Ni40MDYyNSAyMS44NDM3NSA2My4yMzQzNzUgDQpRIDE4LjMxMjUgNjAuMDYyNSAxOC4zMTI1IDU0LjM5MDYyNSANCnoNCiIgaWQ9IkRlamFWdVNhbnMtNTYiLz4NCiAgICAgICA8L2RlZnM+DQogICAgICAgPHVzZSB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OCIvPg0KICAgICAgIDx1c2UgeD0iNjMuNjIzMDQ3IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00NiIvPg0KICAgICAgIDx1c2UgeD0iOTUuNDEwMTU2IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy01NiIvPg0KICAgICAgIDx1c2UgeD0iMTU5LjAzMzIwMyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTAiLz4NCiAgICAgICA8dXNlIHg9IjIyMi42NTYyNSIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTMiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ieXRpY2tfMiI+DQogICAgIDxnIGlkPSJsaW5lMmRfOSI+DQogICAgICA8Zz4NCiAgICAgICA8dXNlIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIgeD0iNDIuODI4MTI1IiB4bGluazpocmVmPSIjbWE4ZmMxZDUwNWYiIHk9IjE3Ny4zMTQ3ODMiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgICA8ZyBpZD0idGV4dF85Ij4NCiAgICAgIDwhLS0gMC44NTAgLS0+DQogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3LjIgMTgxLjExNDAwMilzY2FsZSgwLjEgLTAuMSkiPg0KICAgICAgIDx1c2UgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDgiLz4NCiAgICAgICA8dXNlIHg9IjYzLjYyMzA0NyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDYiLz4NCiAgICAgICA8dXNlIHg9Ijk1LjQxMDE1NiIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTYiLz4NCiAgICAgICA8dXNlIHg9IjE1OS4wMzMyMDMiIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTUzIi8+DQogICAgICAgPHVzZSB4PSIyMjIuNjU2MjUiIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ4Ii8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICA8L2c+DQogICAgPGcgaWQ9Inl0aWNrXzMiPg0KICAgICA8ZyBpZD0ibGluZTJkXzEwIj4NCiAgICAgIDxnPg0KICAgICAgIDx1c2Ugc3R5bGU9InN0cm9rZTojMDAwMDAwO3N0cm9rZS13aWR0aDowLjg7IiB4PSI0Mi44MjgxMjUiIHhsaW5rOmhyZWY9IiNtYThmYzFkNTA1ZiIgeT0iMTUwLjYwOTU5MiIvPg0KICAgICAgPC9nPg0KICAgICA8L2c+DQogICAgIDxnIGlkPSJ0ZXh0XzEwIj4NCiAgICAgIDwhLS0gMC44NzUgLS0+DQogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3LjIgMTU0LjQwODgxMSlzY2FsZSgwLjEgLTAuMSkiPg0KICAgICAgIDxkZWZzPg0KICAgICAgICA8cGF0aCBkPSJNIDguMjAzMTI1IDcyLjkwNjI1IA0KTCA1NS4wNzgxMjUgNzIuOTA2MjUgDQpMIDU1LjA3ODEyNSA2OC43MDMxMjUgDQpMIDI4LjYwOTM3NSAwIA0KTCAxOC4zMTI1IDAgDQpMIDQzLjIxODc1IDY0LjU5Mzc1IA0KTCA4LjIwMzEyNSA2NC41OTM3NSANCnoNCiIgaWQ9IkRlamFWdVNhbnMtNTUiLz4NCiAgICAgICA8L2RlZnM+DQogICAgICAgPHVzZSB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OCIvPg0KICAgICAgIDx1c2UgeD0iNjMuNjIzMDQ3IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00NiIvPg0KICAgICAgIDx1c2UgeD0iOTUuNDEwMTU2IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy01NiIvPg0KICAgICAgIDx1c2UgeD0iMTU5LjAzMzIwMyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTUiLz4NCiAgICAgICA8dXNlIHg9IjIyMi42NTYyNSIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTMiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ieXRpY2tfNCI+DQogICAgIDxnIGlkPSJsaW5lMmRfMTEiPg0KICAgICAgPGc+DQogICAgICAgPHVzZSBzdHlsZT0ic3Ryb2tlOiMwMDAwMDA7c3Ryb2tlLXdpZHRoOjAuODsiIHg9IjQyLjgyODEyNSIgeGxpbms6aHJlZj0iI21hOGZjMWQ1MDVmIiB5PSIxMjMuOTA0NDAxIi8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICAgPGcgaWQ9InRleHRfMTEiPg0KICAgICAgPCEtLSAwLjkwMCAtLT4NCiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDcuMiAxMjcuNzAzNjE5KXNjYWxlKDAuMSAtMC4xKSI+DQogICAgICAgPGRlZnM+DQogICAgICAgIDxwYXRoIGQ9Ik0gMTAuOTg0Mzc1IDEuNTE1NjI1IA0KTCAxMC45ODQzNzUgMTAuNSANClEgMTQuNzAzMTI1IDguNzM0Mzc1IDE4LjUgNy44MTI1IA0KUSAyMi4zMTI1IDYuODkwNjI1IDI1Ljk4NDM3NSA2Ljg5MDYyNSANClEgMzUuNzUgNi44OTA2MjUgNDAuODkwNjI1IDEzLjQ1MzEyNSANClEgNDYuMDQ2ODc1IDIwLjAxNTYyNSA0Ni43ODEyNSAzMy40MDYyNSANClEgNDMuOTUzMTI1IDI5LjIwMzEyNSAzOS41OTM3NSAyNi45NTMxMjUgDQpRIDM1LjI1IDI0LjcwMzEyNSAyOS45ODQzNzUgMjQuNzAzMTI1IA0KUSAxOS4wNDY4NzUgMjQuNzAzMTI1IDEyLjY3MTg3NSAzMS4zMTI1IA0KUSA2LjI5Njg3NSAzNy45Mzc1IDYuMjk2ODc1IDQ5LjQyMTg3NSANClEgNi4yOTY4NzUgNjAuNjQwNjI1IDEyLjkzNzUgNjcuNDIxODc1IA0KUSAxOS41NzgxMjUgNzQuMjE4NzUgMzAuNjA5Mzc1IDc0LjIxODc1IA0KUSA0My4yNjU2MjUgNzQuMjE4NzUgNDkuOTIxODc1IDY0LjUxNTYyNSANClEgNTYuNTkzNzUgNTQuODI4MTI1IDU2LjU5Mzc1IDM2LjM3NSANClEgNTYuNTkzNzUgMTkuMTQwNjI1IDQ4LjQwNjI1IDguODU5Mzc1IA0KUSA0MC4yMzQzNzUgLTEuNDIxODc1IDI2LjQyMTg3NSAtMS40MjE4NzUgDQpRIDIyLjcwMzEyNSAtMS40MjE4NzUgMTguODkwNjI1IC0wLjY4NzUgDQpRIDE1LjA5Mzc1IDAuMDQ2ODc1IDEwLjk4NDM3NSAxLjUxNTYyNSANCnoNCk0gMzAuNjA5Mzc1IDMyLjQyMTg3NSANClEgMzcuMjUgMzIuNDIxODc1IDQxLjEyNSAzNi45NTMxMjUgDQpRIDQ1LjAxNTYyNSA0MS41IDQ1LjAxNTYyNSA0OS40MjE4NzUgDQpRIDQ1LjAxNTYyNSA1Ny4yODEyNSA0MS4xMjUgNjEuODQzNzUgDQpRIDM3LjI1IDY2LjQwNjI1IDMwLjYwOTM3NSA2Ni40MDYyNSANClEgMjMuOTY4NzUgNjYuNDA2MjUgMjAuMDkzNzUgNjEuODQzNzUgDQpRIDE2LjIxODc1IDU3LjI4MTI1IDE2LjIxODc1IDQ5LjQyMTg3NSANClEgMTYuMjE4NzUgNDEuNSAyMC4wOTM3NSAzNi45NTMxMjUgDQpRIDIzLjk2ODc1IDMyLjQyMTg3NSAzMC42MDkzNzUgMzIuNDIxODc1IA0Keg0KIiBpZD0iRGVqYVZ1U2Fucy01NyIvPg0KICAgICAgIDwvZGVmcz4NCiAgICAgICA8dXNlIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ4Ii8+DQogICAgICAgPHVzZSB4PSI2My42MjMwNDciIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ2Ii8+DQogICAgICAgPHVzZSB4PSI5NS40MTAxNTYiIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTU3Ii8+DQogICAgICAgPHVzZSB4PSIxNTkuMDMzMjAzIiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OCIvPg0KICAgICAgIDx1c2UgeD0iMjIyLjY1NjI1IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OCIvPg0KICAgICAgPC9nPg0KICAgICA8L2c+DQogICAgPC9nPg0KICAgIDxnIGlkPSJ5dGlja181Ij4NCiAgICAgPGcgaWQ9ImxpbmUyZF8xMiI+DQogICAgICA8Zz4NCiAgICAgICA8dXNlIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIgeD0iNDIuODI4MTI1IiB4bGluazpocmVmPSIjbWE4ZmMxZDUwNWYiIHk9Ijk3LjE5OTIxIi8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICAgPGcgaWQ9InRleHRfMTIiPg0KICAgICAgPCEtLSAwLjkyNSAtLT4NCiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDcuMiAxMDAuOTk4NDI4KXNjYWxlKDAuMSAtMC4xKSI+DQogICAgICAgPHVzZSB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OCIvPg0KICAgICAgIDx1c2UgeD0iNjMuNjIzMDQ3IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00NiIvPg0KICAgICAgIDx1c2UgeD0iOTUuNDEwMTU2IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy01NyIvPg0KICAgICAgIDx1c2UgeD0iMTU5LjAzMzIwMyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTAiLz4NCiAgICAgICA8dXNlIHg9IjIyMi42NTYyNSIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTMiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ieXRpY2tfNiI+DQogICAgIDxnIGlkPSJsaW5lMmRfMTMiPg0KICAgICAgPGc+DQogICAgICAgPHVzZSBzdHlsZT0ic3Ryb2tlOiMwMDAwMDA7c3Ryb2tlLXdpZHRoOjAuODsiIHg9IjQyLjgyODEyNSIgeGxpbms6aHJlZj0iI21hOGZjMWQ1MDVmIiB5PSI3MC40OTQwMTkiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgICA8ZyBpZD0idGV4dF8xMyI+DQogICAgICA8IS0tIDAuOTUwIC0tPg0KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNy4yIDc0LjI5MzIzNylzY2FsZSgwLjEgLTAuMSkiPg0KICAgICAgIDx1c2UgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDgiLz4NCiAgICAgICA8dXNlIHg9IjYzLjYyMzA0NyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDYiLz4NCiAgICAgICA8dXNlIHg9Ijk1LjQxMDE1NiIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNTciLz4NCiAgICAgICA8dXNlIHg9IjE1OS4wMzMyMDMiIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTUzIi8+DQogICAgICAgPHVzZSB4PSIyMjIuNjU2MjUiIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ4Ii8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICA8L2c+DQogICAgPGcgaWQ9Inl0aWNrXzciPg0KICAgICA8ZyBpZD0ibGluZTJkXzE0Ij4NCiAgICAgIDxnPg0KICAgICAgIDx1c2Ugc3R5bGU9InN0cm9rZTojMDAwMDAwO3N0cm9rZS13aWR0aDowLjg7IiB4PSI0Mi44MjgxMjUiIHhsaW5rOmhyZWY9IiNtYThmYzFkNTA1ZiIgeT0iNDMuNzg4ODI3Ii8+DQogICAgICA8L2c+DQogICAgIDwvZz4NCiAgICAgPGcgaWQ9InRleHRfMTQiPg0KICAgICAgPCEtLSAwLjk3NSAtLT4NCiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDcuMiA0Ny41ODgwNDYpc2NhbGUoMC4xIC0wLjEpIj4NCiAgICAgICA8dXNlIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ4Ii8+DQogICAgICAgPHVzZSB4PSI2My42MjMwNDciIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTQ2Ii8+DQogICAgICAgPHVzZSB4PSI5NS40MTAxNTYiIHhsaW5rOmhyZWY9IiNEZWphVnVTYW5zLTU3Ii8+DQogICAgICAgPHVzZSB4PSIxNTkuMDMzMjAzIiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy01NSIvPg0KICAgICAgIDx1c2UgeD0iMjIyLjY1NjI1IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy01MyIvPg0KICAgICAgPC9nPg0KICAgICA8L2c+DQogICAgPC9nPg0KICAgIDxnIGlkPSJ5dGlja184Ij4NCiAgICAgPGcgaWQ9ImxpbmUyZF8xNSI+DQogICAgICA8Zz4NCiAgICAgICA8dXNlIHN0eWxlPSJzdHJva2U6IzAwMDAwMDtzdHJva2Utd2lkdGg6MC44OyIgeD0iNDIuODI4MTI1IiB4bGluazpocmVmPSIjbWE4ZmMxZDUwNWYiIHk9IjE3LjA4MzYzNiIvPg0KICAgICAgPC9nPg0KICAgICA8L2c+DQogICAgIDxnIGlkPSJ0ZXh0XzE1Ij4NCiAgICAgIDwhLS0gMS4wMDAgLS0+DQogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3LjIgMjAuODgyODU1KXNjYWxlKDAuMSAtMC4xKSI+DQogICAgICAgPHVzZSB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OSIvPg0KICAgICAgIDx1c2UgeD0iNjMuNjIzMDQ3IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00NiIvPg0KICAgICAgIDx1c2UgeD0iOTUuNDEwMTU2IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy00OCIvPg0KICAgICAgIDx1c2UgeD0iMTU5LjAzMzIwMyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDgiLz4NCiAgICAgICA8dXNlIHg9IjIyMi42NTYyNSIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtNDgiLz4NCiAgICAgIDwvZz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgIDwvZz4NCiAgIDxnIGlkPSJsaW5lMmRfMTYiPg0KICAgIDxwYXRoIGNsaXAtcGF0aD0idXJsKCNwMTk4MTYwOWY4YykiIGQ9Ik0gNTguMDQ2MzA3IDIxNC43NTYzNjQgDQpMIDY4LjU0MTYwNSA5OC4xODYwODQgDQpMIDc5LjAzNjkwMiA3Ny40MjE5NzkgDQpMIDg5LjUzMjIgNjQuNjA1ODgxIA0KTCAxMDAuMDI3NDk4IDU1LjU2NTEwMyANCkwgMTEwLjUyMjc5NiA0OC4wNDc2MzQgDQpMIDEyMS4wMTgwOTQgNDMuMTEzMjYyIA0KTCAxMzEuNTEzMzkxIDM4LjUxMDAzOCANCkwgMTQyLjAwODY4OSAzMy41NDI1NTggDQpMIDE1Mi41MDM5ODcgMzAuMjMwODgzIA0KTCAxNjIuOTk5Mjg1IDI4LjYwODE4NiANCkwgMTczLjQ5NDU4MyAyNC42NjczMjMgDQpMIDE4My45ODk4OCAyMy41NDEzOCANCkwgMTk0LjQ4NTE3OCAyMi43Nzk2OTQgDQpMIDIwNC45ODA0NzYgMjAuMzk1MzExIA0KTCAyMTUuNDc1Nzc0IDE5Ljc2NjA1OSANCkwgMjI1Ljk3MTA3MiAxOS42OTk4NDIgDQpMIDIzNi40NjYzNyAxOC4zNzUxODUgDQpMIDI0Ni45NjE2NjcgMTguMjA5NTc5IA0KTCAyNTcuNDU2OTY1IDE3Ljc3OTEwNSANCkwgMjY3Ljk1MjI2MyAxNy43NDU5OTcgDQpMIDI3OC40NDc1NjEgMTcuNDE0Nzg1IA0KTCAyODguOTQyODU5IDE3LjM0ODU2OCANCkwgMjk5LjQzODE1NiAxNy4yMTYxMzQgDQpMIDMwOS45MzM0NTQgMTcuMTgyOTYyIA0KTCAzMjAuNDI4NzUyIDE3LjExNjc0NSANCkwgMzMwLjkyNDA1IDE3LjExNjc0NSANCkwgMzQxLjQxOTM0OCAxNy4xNDk4NTMgDQpMIDM1MS45MTQ2NDUgMTcuMDgzNjM2IA0KTCAzNjIuNDA5OTQzIDE3LjA4MzYzNiANCiIgc3R5bGU9ImZpbGw6bm9uZTtzdHJva2U6IzFmNzdiNDtzdHJva2UtbGluZWNhcDpzcXVhcmU7c3Ryb2tlLXdpZHRoOjEuNTsiLz4NCiAgIDwvZz4NCiAgIDxnIGlkPSJsaW5lMmRfMTciPg0KICAgIDxwYXRoIGNsaXAtcGF0aD0idXJsKCNwMTk4MTYwOWY4YykiIGQ9Ik0gNTguMDQ2MzA3IDExMS42OTkzMDcgDQpMIDY4LjU0MTYwNSA4OC41NjY5OTEgDQpMIDc5LjAzNjkwMiA3Ny45MDA1MjQgDQpMIDg5LjUzMjIgNjcuOTEyMTQ0IA0KTCAxMDAuMDI3NDk4IDYxLjk0MDAxMiANCkwgMTEwLjUyMjc5NiA1Ny44MTk0NzIgDQpMIDEyMS4wMTgwOTQgNTguNjAxODUgDQpMIDEzMS41MTMzOTEgNTAuODU2MzE1IA0KTCAxNDIuMDA4Njg5IDQ3Ljk2MTQ4OCANCkwgMTUyLjUwMzk4NyA1My4xNTEzMDUgDQpMIDE2Mi45OTkyODUgNDcuNjQ4NTQ5IA0KTCAxNzMuNDk0NTgzIDQ1LjM1MzU1OSANCkwgMTgzLjk4OTg4IDQ4LjQ4MzA3NCANCkwgMTk0LjQ4NTE3OCA0Ni41NTMyMzIgDQpMIDIwNC45ODA0NzYgNDYuNzg3OTIgDQpMIDIxNS40NzU3NzQgNDYuMTg4MTQ3IA0KTCAyMjUuOTcxMDcyIDQ1LjMyNzUxOCANCkwgMjM2LjQ2NjM3IDQ0LjM2MjU5NyANCkwgMjQ2Ljk2MTY2NyA0NC4wMjM1NTQgDQpMIDI1Ny40NTY5NjUgNDQuMDc1NyANCkwgMjY3Ljk1MjI2MyA0My41ODAyMTkgDQpMIDI3OC40NDc1NjEgNDMuOTcxNDA4IA0KTCAyODguOTQyODU5IDQzLjA1ODYzMyANCkwgMjk5LjQzODE1NiA0Mi44NDk5ODYgDQpMIDMwOS45MzM0NTQgNDMuNTI4MDA5IA0KTCAzMjAuNDI4NzUyIDQzLjYzMjM2NCANCkwgMzMwLjkyNDA1IDQzLjU4MDIxOSANCkwgMzQxLjQxOTM0OCA0Mi40NTg3OTYgDQpMIDM1MS45MTQ2NDUgNDMuNTAxOTY4IA0KTCAzNjIuNDA5OTQzIDQyLjg0OTk4NiANCiIgc3R5bGU9ImZpbGw6bm9uZTtzdHJva2U6I2ZmN2YwZTtzdHJva2UtbGluZWNhcDpzcXVhcmU7c3Ryb2tlLXdpZHRoOjEuNTsiLz4NCiAgIDwvZz4NCiAgIDxnIGlkPSJwYXRjaF8zIj4NCiAgICA8cGF0aCBkPSJNIDQyLjgyODEyNSAyMjQuNjQgDQpMIDQyLjgyODEyNSA3LjIgDQoiIHN0eWxlPSJmaWxsOm5vbmU7c3Ryb2tlOiMwMDAwMDA7c3Ryb2tlLWxpbmVjYXA6c3F1YXJlO3N0cm9rZS1saW5lam9pbjptaXRlcjtzdHJva2Utd2lkdGg6MC44OyIvPg0KICAgPC9nPg0KICAgPGcgaWQ9InBhdGNoXzQiPg0KICAgIDxwYXRoIGQ9Ik0gMzc3LjYyODEyNSAyMjQuNjQgDQpMIDM3Ny42MjgxMjUgNy4yIA0KIiBzdHlsZT0iZmlsbDpub25lO3N0cm9rZTojMDAwMDAwO3N0cm9rZS1saW5lY2FwOnNxdWFyZTtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLXdpZHRoOjAuODsiLz4NCiAgIDwvZz4NCiAgIDxnIGlkPSJwYXRjaF81Ij4NCiAgICA8cGF0aCBkPSJNIDQyLjgyODEyNSAyMjQuNjQgDQpMIDM3Ny42MjgxMjUgMjI0LjY0IA0KIiBzdHlsZT0iZmlsbDpub25lO3N0cm9rZTojMDAwMDAwO3N0cm9rZS1saW5lY2FwOnNxdWFyZTtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLXdpZHRoOjAuODsiLz4NCiAgIDwvZz4NCiAgIDxnIGlkPSJwYXRjaF82Ij4NCiAgICA8cGF0aCBkPSJNIDQyLjgyODEyNSA3LjIgDQpMIDM3Ny42MjgxMjUgNy4yIA0KIiBzdHlsZT0iZmlsbDpub25lO3N0cm9rZTojMDAwMDAwO3N0cm9rZS1saW5lY2FwOnNxdWFyZTtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLXdpZHRoOjAuODsiLz4NCiAgIDwvZz4NCiAgIDxnIGlkPSJsZWdlbmRfMSI+DQogICAgPGcgaWQ9InBhdGNoXzciPg0KICAgICA8cGF0aCBkPSJNIDMwMS42NzgxMjUgMjE5LjY0IA0KTCAzNzAuNjI4MTI1IDIxOS42NCANClEgMzcyLjYyODEyNSAyMTkuNjQgMzcyLjYyODEyNSAyMTcuNjQgDQpMIDM3Mi42MjgxMjUgMTg5LjAwNTYyNSANClEgMzcyLjYyODEyNSAxODcuMDA1NjI1IDM3MC42MjgxMjUgMTg3LjAwNTYyNSANCkwgMzAxLjY3ODEyNSAxODcuMDA1NjI1IA0KUSAyOTkuNjc4MTI1IDE4Ny4wMDU2MjUgMjk5LjY3ODEyNSAxODkuMDA1NjI1IA0KTCAyOTkuNjc4MTI1IDIxNy42NCANClEgMjk5LjY3ODEyNSAyMTkuNjQgMzAxLjY3ODEyNSAyMTkuNjQgDQp6DQoiIHN0eWxlPSJmaWxsOiNmZmZmZmY7b3BhY2l0eTowLjg7c3Ryb2tlOiNjY2NjY2M7c3Ryb2tlLWxpbmVqb2luOm1pdGVyOyIvPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ibGluZTJkXzE4Ij4NCiAgICAgPHBhdGggZD0iTSAzMDMuNjc4MTI1IDE5NS4xMDQwNjIgDQpMIDMyMy42NzgxMjUgMTk1LjEwNDA2MiANCiIgc3R5bGU9ImZpbGw6bm9uZTtzdHJva2U6IzFmNzdiNDtzdHJva2UtbGluZWNhcDpzcXVhcmU7c3Ryb2tlLXdpZHRoOjEuNTsiLz4NCiAgICA8L2c+DQogICAgPGcgaWQ9ImxpbmUyZF8xOSIvPg0KICAgIDxnIGlkPSJ0ZXh0XzE2Ij4NCiAgICAgPCEtLSBhY2MgLS0+DQogICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMzMS42NzgxMjUgMTk4LjYwNDA2MilzY2FsZSgwLjEgLTAuMSkiPg0KICAgICAgPGRlZnM+DQogICAgICAgPHBhdGggZD0iTSAzNC4yODEyNSAyNy40ODQzNzUgDQpRIDIzLjM5MDYyNSAyNy40ODQzNzUgMTkuMTg3NSAyNSANClEgMTQuOTg0Mzc1IDIyLjUxNTYyNSAxNC45ODQzNzUgMTYuNSANClEgMTQuOTg0Mzc1IDExLjcxODc1IDE4LjE0MDYyNSA4LjkwNjI1IA0KUSAyMS4yOTY4NzUgNi4xMDkzNzUgMjYuNzAzMTI1IDYuMTA5Mzc1IA0KUSAzNC4xODc1IDYuMTA5Mzc1IDM4LjcwMzEyNSAxMS40MDYyNSANClEgNDMuMjE4NzUgMTYuNzAzMTI1IDQzLjIxODc1IDI1LjQ4NDM3NSANCkwgNDMuMjE4NzUgMjcuNDg0Mzc1IA0Keg0KTSA1Mi4yMDMxMjUgMzEuMjAzMTI1IA0KTCA1Mi4yMDMxMjUgMCANCkwgNDMuMjE4NzUgMCANCkwgNDMuMjE4NzUgOC4yOTY4NzUgDQpRIDQwLjE0MDYyNSAzLjMyODEyNSAzNS41NDY4NzUgMC45NTMxMjUgDQpRIDMwLjk1MzEyNSAtMS40MjE4NzUgMjQuMzEyNSAtMS40MjE4NzUgDQpRIDE1LjkyMTg3NSAtMS40MjE4NzUgMTAuOTUzMTI1IDMuMjk2ODc1IA0KUSA2IDguMDE1NjI1IDYgMTUuOTIxODc1IA0KUSA2IDI1LjE0MDYyNSAxMi4xNzE4NzUgMjkuODI4MTI1IA0KUSAxOC4zNTkzNzUgMzQuNTE1NjI1IDMwLjYwOTM3NSAzNC41MTU2MjUgDQpMIDQzLjIxODc1IDM0LjUxNTYyNSANCkwgNDMuMjE4NzUgMzUuNDA2MjUgDQpRIDQzLjIxODc1IDQxLjYwOTM3NSAzOS4xNDA2MjUgNDUgDQpRIDM1LjA2MjUgNDguMzkwNjI1IDI3LjY4NzUgNDguMzkwNjI1IA0KUSAyMyA0OC4zOTA2MjUgMTguNTQ2ODc1IDQ3LjI2NTYyNSANClEgMTQuMTA5Mzc1IDQ2LjE0MDYyNSAxMC4wMTU2MjUgNDMuODkwNjI1IA0KTCAxMC4wMTU2MjUgNTIuMjAzMTI1IA0KUSAxNC45Mzc1IDU0LjEwOTM3NSAxOS41NzgxMjUgNTUuMDQ2ODc1IA0KUSAyNC4yMTg3NSA1NiAyOC42MDkzNzUgNTYgDQpRIDQwLjQ4NDM3NSA1NiA0Ni4zNDM3NSA0OS44NDM3NSANClEgNTIuMjAzMTI1IDQzLjcwMzEyNSA1Mi4yMDMxMjUgMzEuMjAzMTI1IA0Keg0KIiBpZD0iRGVqYVZ1U2Fucy05NyIvPg0KICAgICAgIDxwYXRoIGQ9Ik0gNDguNzgxMjUgNTIuNTkzNzUgDQpMIDQ4Ljc4MTI1IDQ0LjE4NzUgDQpRIDQ0Ljk2ODc1IDQ2LjI5Njg3NSA0MS4xNDA2MjUgNDcuMzQzNzUgDQpRIDM3LjMxMjUgNDguMzkwNjI1IDMzLjQwNjI1IDQ4LjM5MDYyNSANClEgMjQuNjU2MjUgNDguMzkwNjI1IDE5LjgxMjUgNDIuODQzNzUgDQpRIDE0Ljk4NDM3NSAzNy4zMTI1IDE0Ljk4NDM3NSAyNy4yOTY4NzUgDQpRIDE0Ljk4NDM3NSAxNy4yODEyNSAxOS44MTI1IDExLjczNDM3NSANClEgMjQuNjU2MjUgNi4yMDMxMjUgMzMuNDA2MjUgNi4yMDMxMjUgDQpRIDM3LjMxMjUgNi4yMDMxMjUgNDEuMTQwNjI1IDcuMjUgDQpRIDQ0Ljk2ODc1IDguMjk2ODc1IDQ4Ljc4MTI1IDEwLjQwNjI1IA0KTCA0OC43ODEyNSAyLjA5Mzc1IA0KUSA0NS4wMTU2MjUgMC4zNDM3NSA0MC45ODQzNzUgLTAuNTMxMjUgDQpRIDM2Ljk2ODc1IC0xLjQyMTg3NSAzMi40MjE4NzUgLTEuNDIxODc1IA0KUSAyMC4wNjI1IC0xLjQyMTg3NSAxMi43ODEyNSA2LjM0Mzc1IA0KUSA1LjUxNTYyNSAxNC4xMDkzNzUgNS41MTU2MjUgMjcuMjk2ODc1IA0KUSA1LjUxNTYyNSA0MC42NzE4NzUgMTIuODU5Mzc1IDQ4LjMyODEyNSANClEgMjAuMjE4NzUgNTYgMzMuMDE1NjI1IDU2IA0KUSAzNy4xNTYyNSA1NiA0MS4xMDkzNzUgNTUuMTQwNjI1IA0KUSA0NS4wNjI1IDU0LjI5Njg3NSA0OC43ODEyNSA1Mi41OTM3NSANCnoNCiIgaWQ9IkRlamFWdVNhbnMtOTkiLz4NCiAgICAgIDwvZGVmcz4NCiAgICAgIDx1c2UgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtOTciLz4NCiAgICAgIDx1c2UgeD0iNjEuMjc5Mjk3IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy05OSIvPg0KICAgICAgPHVzZSB4PSIxMTYuMjU5NzY2IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy05OSIvPg0KICAgICA8L2c+DQogICAgPC9nPg0KICAgIDxnIGlkPSJsaW5lMmRfMjAiPg0KICAgICA8cGF0aCBkPSJNIDMwMy42NzgxMjUgMjA5Ljc4MjE4NyANCkwgMzIzLjY3ODEyNSAyMDkuNzgyMTg3IA0KIiBzdHlsZT0iZmlsbDpub25lO3N0cm9rZTojZmY3ZjBlO3N0cm9rZS1saW5lY2FwOnNxdWFyZTtzdHJva2Utd2lkdGg6MS41OyIvPg0KICAgIDwvZz4NCiAgICA8ZyBpZD0ibGluZTJkXzIxIi8+DQogICAgPGcgaWQ9InRleHRfMTciPg0KICAgICA8IS0tIHZhbF9hY2MgLS0+DQogICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMzMS42NzgxMjUgMjEzLjI4MjE4NylzY2FsZSgwLjEgLTAuMSkiPg0KICAgICAgPGRlZnM+DQogICAgICAgPHBhdGggZD0iTSAyLjk4NDM3NSA1NC42ODc1IA0KTCAxMi41IDU0LjY4NzUgDQpMIDI5LjU5Mzc1IDguNzk2ODc1IA0KTCA0Ni42ODc1IDU0LjY4NzUgDQpMIDU2LjIwMzEyNSA1NC42ODc1IA0KTCAzNS42ODc1IDAgDQpMIDIzLjQ4NDM3NSAwIA0Keg0KIiBpZD0iRGVqYVZ1U2Fucy0xMTgiLz4NCiAgICAgICA8cGF0aCBkPSJNIDkuNDIxODc1IDc1Ljk4NDM3NSANCkwgMTguNDA2MjUgNzUuOTg0Mzc1IA0KTCAxOC40MDYyNSAwIA0KTCA5LjQyMTg3NSAwIA0Keg0KIiBpZD0iRGVqYVZ1U2Fucy0xMDgiLz4NCiAgICAgICA8cGF0aCBkPSJNIDUwLjk4NDM3NSAtMTYuNjA5Mzc1IA0KTCA1MC45ODQzNzUgLTIzLjU3ODEyNSANCkwgLTAuOTg0Mzc1IC0yMy41NzgxMjUgDQpMIC0wLjk4NDM3NSAtMTYuNjA5Mzc1IA0Keg0KIiBpZD0iRGVqYVZ1U2Fucy05NSIvPg0KICAgICAgPC9kZWZzPg0KICAgICAgPHVzZSB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy0xMTgiLz4NCiAgICAgIDx1c2UgeD0iNTkuMTc5Njg4IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy05NyIvPg0KICAgICAgPHVzZSB4PSIxMjAuNDU4OTg0IiB4bGluazpocmVmPSIjRGVqYVZ1U2Fucy0xMDgiLz4NCiAgICAgIDx1c2UgeD0iMTQ4LjI0MjE4OCIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtOTUiLz4NCiAgICAgIDx1c2UgeD0iMTk4LjI0MjE4OCIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtOTciLz4NCiAgICAgIDx1c2UgeD0iMjU5LjUyMTQ4NCIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtOTkiLz4NCiAgICAgIDx1c2UgeD0iMzE0LjUwMTk1MyIgeGxpbms6aHJlZj0iI0RlamFWdVNhbnMtOTkiLz4NCiAgICAgPC9nPg0KICAgIDwvZz4NCiAgIDwvZz4NCiAgPC9nPg0KIDwvZz4NCiA8ZGVmcz4NCiAgPGNsaXBQYXRoIGlkPSJwMTk4MTYwOWY4YyI+DQogICA8cmVjdCBoZWlnaHQ9IjIxNy40NCIgd2lkdGg9IjMzNC44IiB4PSI0Mi44MjgxMjUiIHk9IjcuMiIvPg0KICA8L2NsaXBQYXRoPg0KIDwvZGVmcz4NCjwvc3ZnPg0K"  />

#### Step6: 在测试集上评估

评估测试集上的损失和准确率。

```python
model.evaluate(x_test, y_test)
```

```txt
{'loss': 0.121063925, 'acc': 0.9736}
```



## 社群交流

如果您在使用中遇到问题，可通过如下方式获取支持：

+ 提交 [Github Issue](https://github.com/blueloveTH/keras4torch/issues) 
+ 发送邮件至 blueloveTH@foxmail.com 或 zhangzhipengcs@foxmail.com



## 贡献

如果您有任何的想法和建议，请随时和我们联系，您的想法对我们非常重要。

同时也欢迎您加入我们，一同维护这个项目。



