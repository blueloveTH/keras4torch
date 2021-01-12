# Utilities

#### `utils.to_tensor(*args)`

Convert the parameter list to `torch.Tensor`.



#### `utils.data.SlicedDataset(slice, *array)`

Create a sliced dataset. It just keeps a reference of `array` thus can avoid memory copy.

+   `slice` (1D-array or list): The slice sequence.
+   `array` (`ndarray` or `torch.Tensor`): Your data.