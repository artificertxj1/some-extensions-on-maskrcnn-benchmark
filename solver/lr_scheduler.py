from bisect import bisect_right
import numpy as np
import torch

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            max_iter,
            lr_min,
            gamma=0.1,
            warmup_factor=1.0/3,
            warmup_iters=500,
            warmup_method="linear",
            decay_method="step",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increased integers"
            )
        if warmup_method  not in ("constant", "linear"):
            raise ValueError(
                "Only constant or linear warmup method accepted"
            )
        assert(decay_method in ("step", "cosine")), \
            "decay_method can only be in (step, cosine) get {} instead".format(decay_method)

        assert(isinstance(max_iter, int)), "max_iter must be an integer"

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.decay_method = decay_method
        self.max_iter = max_iter
        self.lr_min = lr_min
        super(WarmupMultiStepLR, self). __init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch  < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        if self.decay_method == "step":

            return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                (self.lr_min +
                  (base_lr - self.lr_min) *
                  (1. + np.cos(np.pi * self.last_epoch / self.max_iter)) / 2.0) * warmup_factor
                for base_lr in self.base_lrs
            ]