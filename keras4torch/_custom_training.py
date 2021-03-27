class TrainerLoopConfig():
    def __init__(self):
        self._train = None

    @property
    def training(self) -> bool:
        return self._train

    def process_batch(self, batch):
        *x_batch, y_batch = batch
        return x_batch, y_batch

    def forward_call(self, model, x_batch):
        return model(*x_batch)

    def prepare_for_optimizer_step(self, model):
        pass

    def prepare_for_metrics_update(self, y_batch_pred, y_batch):
        return y_batch_pred, y_batch

    def cache_for_epoch_metrics(self, y_batch_pred, y_batch):
        return y_batch_pred.cpu(), y_batch.cpu()


_default_trainer_loop_config = TrainerLoopConfig()