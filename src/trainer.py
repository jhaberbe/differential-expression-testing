import torch
from tqdm import tqdm

class LossSlopeLRScheduler:
    """
    Adaptive LR controller based on loss change (slope) + optional ReduceLROnPlateau.
    - If |Δloss| < delta_low: shrink LR by decrease_factor
    - If |Δloss| > delta_high: grow LR by increase_factor
    - Always clamps to [min_lr, max_lr]
    - Optionally wraps a torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
        delta_low: float = 1e-3,
        delta_high: float = 1e-1,
        decrease_factor: float = 0.9,
        increase_factor: float = 1.1,
        min_lr: float = 1e-5,
        max_lr: float = 1.0,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
    ):
        self.opt = optimizer
        self.plateau = plateau_scheduler
        self.delta_low = float(delta_low)
        self.delta_high = float(delta_high)
        self.decrease_factor = float(decrease_factor)
        self.increase_factor = float(increase_factor)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.warmup_steps = int(warmup_steps)
        self.cooldown_steps = int(cooldown_steps)

        self._prev_loss = None
        self._cooldown = 0
        self._step_idx = 0

    def _get_lrs(self):
        return [g["lr"] for g in self.opt.param_groups]

    def _set_lrs(self, new_lrs):
        for g, lr in zip(self.opt.param_groups, new_lrs):
            g["lr"] = float(lr)

    def _scale_all(self, factor):
        new_lrs = []
        for lr in self._get_lrs():
            lr = lr * factor
            lr = max(self.min_lr, min(self.max_lr, lr))
            new_lrs.append(lr)
        self._set_lrs(new_lrs)

    def step(self, current_loss: float):
        # First, let ReduceLROnPlateau observe the metric (if provided)
        if self.plateau is not None:
            self.plateau.step(current_loss)

        # Warmup: just record the loss for a few steps
        self._step_idx += 1
        if self._step_idx <= self.warmup_steps:
            self._prev_loss = current_loss
            return

        # Cooldown handling (avoid too-frequent LR flaps)
        if self._cooldown > 0:
            self._cooldown -= 1
            self._prev_loss = current_loss
            return

        # Slope-based decision
        if self._prev_loss is not None:
            delta = abs(current_loss - self._prev_loss)
            if delta < self.delta_low:
                self._scale_all(self.decrease_factor)
                self._cooldown = self.cooldown_steps
            elif delta > self.delta_high:
                self._scale_all(self.increase_factor)
                self._cooldown = self.cooldown_steps

        self._prev_loss = current_loss


class Trainer:
    """
    Minimal trainer that uses a user-provided `loss_fn()` closure returning a scalar torch.Tensor.
    Handles: zero_grad -> backward -> step -> scheduler.step(loss)
    """
    def __init__(self, optimizer, scheduler: LossSlopeLRScheduler, loss_fn):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

    def fit(self, epochs=100, show_tqdm=True):
        iterator = range(epochs)
        if show_tqdm:
            iterator = tqdm(iterator)

        for epoch in iterator:
            self.optimizer.zero_grad()
            loss = self.loss_fn()           # returns a scalar tensor
            loss.backward()
            self.optimizer.step()

            loss_val = float(loss.item())
            self.scheduler.step(loss_val)

            if show_tqdm:
                # show the first group's LR (typical single-group case)
                iterator.set_postfix(
                    loss=loss_val,
                    lr=self.optimizer.param_groups[0]["lr"]
                )