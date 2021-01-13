class TrainerLoopConfig():
    def __init__(self) -> None:
        super().__init__()
        self.train = None

    def process_batch(self, x_batch, y_batch):
        return x_batch, y_batch

    def forward_call(self, model, x_batch):
        return model(*x_batch)

    def prepare_for_optimizer_step(self, model):
        pass

    def prepare_for_metrics_update(self, y_batch_pred, y_batch):
        return y_batch_pred, y_batch


_default_trainer_loop_config = TrainerLoopConfig()