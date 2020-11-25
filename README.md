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

<img src="docs/learning_curve.svg"  />

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



