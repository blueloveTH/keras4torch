# Metrics

A metric is a function that is used to judge the performance of your model.

Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any loss function as a metric.



## Available metrics

### For classification

+ Accuracy
+ categorical_accuracy
+ binary_accuracy
+ ROC_AUC
+ F1_Score

### For regression

+ MeanSquaredError
+ MeanAbsoluteError
+ RootMeanSquaredError



## Custom metrics

Any callable object can be custom metrics.

For example,

```python
def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
```

Or

```python
class MSE(object):
    def __call__(y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)
```



If you want to name the metric with an abbr. or alias, you can subclass `keras4torch.metrics.Metric` and implement `get_abbr()` method. Another way to do the same thing is providing a `dict` to `compile()` when you config the model.

```python
model.compile(
    optimizer='adam',
    loss='mae',
    metrics={'mse': MeanSquaredError()}
)
```

Then the model will use `'mse'` as the abbr. of `MeanSquaredError` in the logger.

