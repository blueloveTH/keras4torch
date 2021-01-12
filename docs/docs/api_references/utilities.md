# Utilities

#### `utils.to_tensor(*args)`

Convert the parameter list to `torch.Tensor`.



#### `utils.data.SlicedDataset(slice, *array)`

Create a sliced dataset without memory copy.

+   `slice` (1D-array or list): Your slice.
+   `array` (`ndarray` or `torch.Tensor`): Your data.