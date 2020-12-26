# Losses

#### CELoss

Here we provide an alternative to `nn.CrossEntropy`, which integrates label smoothing.

```python
loss_fn = keras4torch.losses.CELoss(label_smoothing=0.1)
```



