# Keras4Torch

#### "An Easy-to-Use Wrapper for Training PyTorch Models❤"

#### “开箱即用”的PyTorch模型训练高级API

[![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/keras4torch.svg)](https://pypi.python.org/pypi/keras4torch)
[![License](https://img.shields.io/github/license/blueloveTH/keras4torch.svg)](https://github.com/blueloveTH/keras4torch/blob/master/LICENSE)



## 安装与配置

```
pip install keras4torch
```

Keras4Torch 支持 Python 3.6 及以上版本.



## 快速开始

Let's start with a simple example of MNIST!

```python
import torch
import torchvision
from torch import nn

import keras4torch
```

#### Step1: 数据预处理

```python
mnist = torchvision.datasets.MNIST(root='./', download=True)
X, y = mnist.train_data, mnist.train_labels

X = X.float() / 255.0    # 将像素缩放到 [0, 1]

x_train, y_train = X[:40000], y[:40000]
x_test, y_test = X[40000:], y[40000:]
```

#### Step2: 构建模型

```python
model = torch.nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 512), nn.ReLU(),
    nn.Linear(512, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

model = keras4torch.Model(model)    # 封装torch模块以集成训练API
```

#### Step3: 设置优化器、损失函数和度量

```python
model.compile(optimizer='adam', loss=nn.CrossEntropyLoss(), metrics=['acc'])
```

#### Step4: 训练模型

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

```
history.plot(kind='line', y=['acc', 'val_acc'])
```

<img src="docs/learning_curve.svg"  />

#### Step6: 在测试集上验证

```python
model.evaluate(x_test, y_test)
```

```txt
OrderedDict([('loss', 0.121063925), ('acc', 0.9736)])
```



## 社群交流

如果您在使用中遇到问题，可通过如下方式获取支持：

+ 提交 [Github Issue](https://github.com/blueloveTH/keras4torch/issues) 
+ 发送邮件给 blueloveTH@foxmail.com 或 zhangzhipengcs@foxmail.com.



