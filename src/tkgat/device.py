"""Device selection and seeding shared by every trainer.

Every model and trainer takes a ``device`` string: ``"auto"`` (CUDA when available, else CPU),
``"cuda"``, ``"cuda:1"`` or ``"cpu"``.  Results are reproducible for a given (seed, device); the same
seed on CPU and on GPU gives different random streams and therefore slightly different numbers, so the
device is written to ``results/registry.csv`` with every run.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def resolve_device(name: str | torch.device = "auto") -> torch.device:
    if isinstance(name, torch.device):
        return name
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False; "
                           "install a CUDA build of torch or pass --device cpu")
    return dev


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch (CPU and every GPU).

    ``deterministic`` asks cuBLAS and torch for deterministic kernels so that two GPU runs with the same
    seed give the same numbers.  It costs a little speed; kernels without a deterministic version only warn.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def describe(device: torch.device) -> str:
    """Short label for the registry, e.g. ``cuda:NVIDIA GeForce RTX 3050 Laptop GPU`` or ``cpu``."""
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return "cpu"
