# Training API

By default, the training pipeline of `keras4torch` can handle many useful cases. While in specific situation, you may want to customize it. To this end, we provide some hooks.

## Get loop configs after `.compile()`

```python
trn_loop = model.trainer.batch_training_loop
val_loop = model.trainer.batch_validation_loop
```

Both `trn_loop` and `val_loop` are python `dict` which contains several hooks of the training pipeline.

Set values of them before fitting or evaluating the model.

