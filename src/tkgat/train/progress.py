"""Atomic epoch checkpoints. Payloads contain tensors/primitives for weights-only loading."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def save_epoch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    torch.save(payload, pending)
    os.replace(pending, path)
    history = payload["history"]
    csv_path = path.with_suffix(".csv")
    pending_csv = csv_path.with_name(csv_path.name + ".tmp")
    with pending_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in history for key in row)))
        writer.writeheader()
        writer.writerows(history)
    os.replace(pending_csv, csv_path)


def load_epoch(path: Path, signature: str, device: torch.device) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("signature") != signature:
        raise ValueError("Epoch checkpoint belongs to another configuration, seed, graph or split")
    if saved["device_type"] != device.type:
        raise ValueError("Epoch resume must use the same device type to restore sampler RNG state")
    return saved
