class TrainerLoopConfig(object):
    def __init__(self) -> None:
        super().__init__()
        self.train = None

    def process_batch(self, batch, device):
        for i in range(len(batch)):
            batch[i] = batch[i].to(device=device)
        return batch[:-1], batch[-1]

    def forward_call(self, model, x_batch, y_batch):
        y_batch_pred = model(*x_batch)
        return y_batch_pred, y_batch

    def prepare_for_optimizer_step(self, model):
        pass

    def prepare_for_metrics_update(self, y_batch_pred, y_batch):
        return y_batch_pred, y_batch


_default_trainer_loop_config = TrainerLoopConfig()