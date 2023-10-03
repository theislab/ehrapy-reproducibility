from torch.optim import lr_scheduler
from enum import Enum
import pytorch_lightning as pl
import os
from torch.optim.lr_scheduler import CyclicLR


def get_scheduler(optimizer, lr_policy, n_ep_decay, n_ep, init_lr):
    if lr_policy == "lambda":

        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - n_ep_decay) / float(n_ep - n_ep_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=n_ep_decay, gamma=0.1)

    elif lr_policy == "cyclic":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=n_ep_decay,  # Number of iterations for the first restart
            T_mult=1,  # A factor increases TiTi​ after a restart
            eta_min=init_lr,
        )  # Minimum learning rate

    elif lr_policy == "constant":
        scheduler = None

    else:
        return NotImplementedError("no such learn rate policy")
    return scheduler


class NNModel(Enum):
    SpatialSSL = "SpatialSSL"
    ResNet18 = "ResNet18"
    # ODE_LSTM = "ODE_LSTM"
    # LSTM ="LSTM"
    # TabNet = "TabNet"
    # GRU = "GRU"


class Stage(Enum):
    fit = "fit"
    test = "test"
    predict = "predict"


class FinishCallback(pl.Callback):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        # self.fold_ind = fold_ind

    def on_train_end(self, trainer, pl_module):
        with open(os.path.join(self.model_path, "status.txt"), "w") as f:
            f.write(f"Status: job was finished")
