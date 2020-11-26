# Callbacks API

A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc).

You can use callbacks to:

- Write TensorBoard logs after every batch of training to monitor your metrics
- Periodically save your model to disk
- Do early stopping
- Get a view on internal states and statistics of a model during training
- ...and more



## Usage of callbacks via the built-in `fit()` loop

You can pass a list of callbacks (as the keyword argument `callbacks`) to the `.fit()` method of a model:

```
my_callbacks = [
    k4t.callbacks.EarlyStopping(patience=2),
    k4t.callbacks.ModelCheckpoint('best_model.pt', monitor='val_acc'),
]
model.fit(x, y, epochs=10, callbacks=my_callbacks)
```

The relevant methods of the callbacks will then be called at each stage of the training.



## Available callbacks

+ ModelCheckpoint
+ EarlyStopping
+ LRScheduler
+ LambdaCallback



## Custom callbacks

Using `LambdaCallback`.