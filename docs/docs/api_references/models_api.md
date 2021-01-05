# Models API

`keras4torch.Model` wraps a `torch.nn.Module` to integrate training and inference features.

#### Configs

##### `.compile(optimizer, loss, metrics=None, epoch_metrics=None, device=None)`

Configure the model for training.

+ `optimizer`: String (name of optimizer) or optimizer instance.

+  `loss`: String (name of objective function), objective function or loss instance.

+ `metrics`: List of metrics to be evaluated by the model during training. You can also use dict to specify the abbreviation of each metric.

+ `epoch_metrics`: List of non-linear metrics(e.g. ROC_AUC) that need to be evaluated on epoch end.

+ `device`: Device of the model and its trainer, if `None` 'cuda' will be used when `torch.cuda.is_available()` otherwise 'cpu'.



#### NumPy workflow

##### `.fit(x, y, epochs, batch_size=32, ...)`

Train the model for a fixed number of epochs (iterations on a dataset).

+ `x` (`ndarray` or `torch.Tensor` or list): Input data

+ `y` (`ndarray` or `torch.Tensor`): Target data

+ `epochs` (int): Number of epochs to train the model

+ `batch_size` (int, default=32): Number of samples per gradient update

+ `validation_batch_size` (int, default=None): Number of samples for each step on validation loop, if `None` will use `batch_size`

+ `validation_split` (float between 0 and 1): Fraction of the training data to be used as validation data

+ `shuffle_val_split` (bool, default=True): Whether to do shuffling when `validation_split` is provided

+ `validation_data` (tuple of `x` and `y`): Data on which to evaluate the loss and any model metrics at the end of each epoch
  
+ `callbacks` (list of `keras4torch.callbacks.Callback`): List of callbacks to apply during training

+ `verbose` (int, default=1): 0, 1, or 2. Verbosity mode. 0 = silent, 1 = normal, 2 = brief

+ `shuffle` (bool, default=True): Whether to shuffle the training data before each epoch

+ `sample_weight` (list of floats): Optional weights for the training samples. If provided will enable `WeightedRandomSampler`
  
+ `num_workers` (int, default=0): Workers of `DataLoader`. If `-1` will use `cpu_count() - 1` for multiprocessing

+ `use_amp` (bool, default=False): Whether to use automatic mixed precision

##### `.evaluate(x, y, batch_size=32, ...)`



##### `.predict(x, batch_size=32, ...)`



#### DataLoader workflow

##### `.fit_dl(train_loader, epochs, val_loader=None, ...)`



##### `.evaluate_dl(data_loader, ...)`



##### `.predict_dl(data_loader, ...)`





#### Saving & Serialization

##### `.save_weights(filepath)`

Equal to `torch.save(model.state_dict(), filepath)`.

##### `.load_weights(filepath)`

Equal to `model.load_state_dict(torch.load(filepath))`.



#### Utilities

##### `.summary(depth=3, ...)`

Print a string summary of the network.

+   `depth` (default=3): Summary details level

##### `.count_params()`

Count the total number of scalars composing the weights.