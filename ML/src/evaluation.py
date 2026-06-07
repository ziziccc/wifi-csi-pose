from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data import CachedWindowDataset


def build_window_loader(
    npz_path: str | Path,
    *,
    window_size: int,
    window_stride: int,
    batch_size: int,
    feature_mode: str,
    require_full_window_mask: bool,
    fill_mode: str,
    max_gap: int,
    limit_windows: int | None = None,
    num_workers: int = 0,
) -> tuple[CachedWindowDataset | Subset, DataLoader]:
    dataset = CachedWindowDataset(
        npz_path=npz_path,
        window_size=window_size,
        window_stride=window_stride,
        feature_mode=feature_mode,
        require_full_window_mask=require_full_window_mask,
        fill_mode=fill_mode,
        max_gap=max_gap,
        return_prev_target=False,
        return_file_id=False,
        motion_lag=1,
    )
    limited_dataset: CachedWindowDataset | Subset = dataset
    if limit_windows is not None and limit_windows > 0 and len(dataset) > limit_windows:
        limited_dataset = Subset(dataset, list(range(int(limit_windows))))
    loader = DataLoader(limited_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return limited_dataset, loader


def evaluate_pose_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float | int]:
    model.eval()
    model.to(device)
    criterion = nn.SmoothL1Loss(reduction="sum")
    total_loss = 0.0
    total_abs_error = 0.0
    total_values = 0
    total_windows = 0
    started_at = time.perf_counter()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            prediction = model(inputs)
            total_loss += float(criterion(prediction, targets).item())
            total_abs_error += float(torch.abs(prediction - targets).sum().item())
            total_values += int(targets.numel())
            total_windows += int(targets.shape[0])

    elapsed_sec = time.perf_counter() - started_at
    return {
        "windows": total_windows,
        "loss": total_loss / max(total_values, 1),
        "mae_norm": total_abs_error / max(total_values, 1),
        "elapsed_sec": elapsed_sec,
        "windows_per_sec": total_windows / elapsed_sec if elapsed_sec > 0 else 0.0,
    }


def checkpoint_data_settings(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {}) or {}
    return {
        "window_size": int(config.get("window_size", 10)),
        "window_stride": int(config.get("window_stride", 1)),
        "feature_mode": str(config.get("feature_mode", "all")),
        "require_full_window_mask": bool(config.get("require_full_window_mask", False)),
        "fill_mode": str(config.get("fill_mode", "forward_fill")),
        "max_gap": int(config.get("max_gap", 3)),
    }
