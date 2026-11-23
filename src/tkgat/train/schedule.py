"""Learning-rate schedules shared by every trainer (decision 0016 item 5).

One schedule for all models, set from the epoch number only (no validation score, no test score is read):

* ``none``   the constant rate of decision 0013 (the default, so every older run is reproduced exactly);
* ``cosine`` cosine decay from the grid learning rate ``lr0`` at epoch 1 to ``lr0 * final_ratio`` at epoch ``horizon``,
  ``lr(e) = lr0 * (r + (1 - r) * (1 + cos(pi * (e - 1) / (horizon - 1))) / 2)`` with ``r = final_ratio``.

``horizon`` is the trainer's ``max_epochs``. A refit (decision 0016 item 4) trains for fewer epochs than ``max_epochs``
and keeps the same horizon, so at epoch ``E`` it has the learning rate the selected state had at epoch ``E``.
"""

from __future__ import annotations

import math

SCHEDULES = ("none", "cosine")


def check_schedule(schedule: str, final_ratio: float) -> None:
    if schedule not in SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {SCHEDULES}, got {schedule!r}")
    if not 0 < final_ratio <= 1:
        raise ValueError(f"lr_final_ratio must be in (0, 1], got {final_ratio}")


def lr_at(epoch: int, base_lr: float, schedule: str, horizon: int, final_ratio: float = 0.01) -> float:
    """Learning rate of ``epoch`` (1-based) under ``schedule``; epochs after ``horizon`` keep the final rate."""
    check_schedule(schedule, final_ratio)
    if epoch < 1:
        raise ValueError(f"epoch is 1-based, got {epoch}")
    if schedule == "none" or horizon <= 1:
        return base_lr
    progress = min(epoch - 1, horizon - 1) / (horizon - 1)
    return base_lr * (final_ratio + (1 - final_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def set_lr(optimizers, lr: float) -> None:
    """Write ``lr`` into every parameter group of the given optimizers."""
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = lr
