def process_batch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device=device)
    return batch[:-1], batch[-1]

def forward_call(model, x_batch, y_batch):
    y_batch_pred = model(*x_batch)
    return y_batch_pred, y_batch

def update_metrics(metrics_rec, y_batch_pred, y_batch):
    metrics_rec.update(y_batch_pred, y_batch)

def create_batch_training_loop():
    return {i.__name__:i for i in [process_batch, forward_call, update_metrics]}

def create_batch_validation_loop():
    return {i.__name__:i for i in [process_batch, forward_call, update_metrics]}

