from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from data import CachedWindowDataset
from original_models import build_model


@dataclass
class TrainConfig:
    cache_dir: Path
    output_dir: Path
    model_name: str
    window_size: int
    window_stride: int
    feature_mode: str
    require_full_window_mask: bool
    fill_mode: str
    max_gap: int
    train_split_name: str
    val_split_name: str
    test_split_name: str
    limit_train_windows: int | None
    limit_val_windows: int | None
    limit_test_windows: int | None
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    hidden_dim: int
    dropout: float
    device: str
    log_interval: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    resume_from_last_checkpoint: bool
    resume_checkpoint_path: Path | None
    init_checkpoint_path: Path | None


def _resolve_config_path(base_dir: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _load_config(config_path: Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    config_path = config_path.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if overrides:
        payload.update({key: value for key, value in overrides.items() if value is not None})

    model_name = str(payload.get("model_name", "multi_esp_fast_pose_cnn"))
    if model_name != "multi_esp_fast_pose_cnn":
        raise ValueError(f"ML2 only supports model_name='multi_esp_fast_pose_cnn', received {model_name!r}.")

    config_dir = config_path.parent
    return TrainConfig(
        cache_dir=_resolve_config_path(config_dir, payload["cache_dir"]),
        output_dir=_resolve_config_path(config_dir, payload["output_dir"]),
        model_name=model_name,
        window_size=int(payload.get("window_size", 10)),
        window_stride=int(payload.get("window_stride", 1)),
        feature_mode=str(payload.get("feature_mode", "all")),
        require_full_window_mask=bool(payload.get("require_full_window_mask", False)),
        fill_mode=str(payload.get("fill_mode", "forward_fill")),
        max_gap=int(payload.get("max_gap", 3)),
        train_split_name=str(payload.get("train_split_name", "train")),
        val_split_name=str(payload.get("val_split_name", "val")),
        test_split_name=str(payload.get("test_split_name", "test")),
        limit_train_windows=(
            int(payload["limit_train_windows"]) if payload.get("limit_train_windows") is not None else None
        ),
        limit_val_windows=(int(payload["limit_val_windows"]) if payload.get("limit_val_windows") is not None else None),
        limit_test_windows=(
            int(payload["limit_test_windows"]) if payload.get("limit_test_windows") is not None else None
        ),
        batch_size=int(payload.get("batch_size", 64)),
        epochs=int(payload.get("epochs", 25)),
        learning_rate=float(payload.get("learning_rate", 1e-3)),
        weight_decay=float(payload.get("weight_decay", 1e-4)),
        num_workers=int(payload.get("num_workers", 0)),
        seed=int(payload.get("seed", 42)),
        hidden_dim=int(payload.get("hidden_dim", 128)),
        dropout=float(payload.get("dropout", 0.15)),
        device=str(payload.get("device", "auto")),
        log_interval=int(payload.get("log_interval", 50)),
        early_stopping_patience=int(payload.get("early_stopping_patience", 5)),
        early_stopping_min_delta=float(payload.get("early_stopping_min_delta", 0.0)),
        resume_from_last_checkpoint=bool(payload.get("resume_from_last_checkpoint", False)),
        resume_checkpoint_path=(
            _resolve_config_path(config_dir, payload["resume_checkpoint_path"])
            if payload.get("resume_checkpoint_path")
            else None
        ),
        init_checkpoint_path=(
            _resolve_config_path(config_dir, payload["init_checkpoint_path"])
            if payload.get("init_checkpoint_path")
            else None
        ),
    )


def _serialize_checkpoint_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_checkpoint_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_checkpoint_value(item) for item in value]
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def _configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _finalize_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], DataLoader]:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": bool(num_workers > 0),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    return dataset, DataLoader(dataset, **loader_kwargs)


def _limit_dataset_windows(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    limit_windows: int | None,
) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    if limit_windows is None or limit_windows <= 0 or len(dataset) <= limit_windows:
        return dataset
    return Subset(dataset, list(range(limit_windows)))


def _build_loader(
    npz_path: Path,
    config: TrainConfig,
    *,
    shuffle: bool,
    num_workers: int | None = None,
    limit_windows: int | None = None,
) -> tuple[Dataset[tuple[torch.Tensor, torch.Tensor]], DataLoader]:
    dataset = CachedWindowDataset(
        npz_path=npz_path,
        window_size=config.window_size,
        window_stride=config.window_stride,
        feature_mode=config.feature_mode,
        require_full_window_mask=config.require_full_window_mask,
        fill_mode=config.fill_mode,
        max_gap=config.max_gap,
        return_prev_target=False,
        return_file_id=False,
        motion_lag=1,
    )
    dataset = _limit_dataset_windows(dataset, limit_windows)
    return _finalize_loader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers if num_workers is None else num_workers,
        shuffle=shuffle,
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_trusted_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _checkpoint_config_value(checkpoint: dict[str, Any], key: str) -> Any:
    config = checkpoint.get("config")
    return config.get(key) if isinstance(config, dict) else None


def _save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    config: TrainConfig,
    in_channels: int,
    epoch: int,
    best_val_loss: float,
    best_train_loss: float,
    best_epoch: int,
    best_train_epoch: int,
    epochs_without_improvement: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": _serialize_checkpoint_value(config.__dict__),
            "model_name": config.model_name,
            "input_channels": in_channels,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_train_loss": best_train_loss,
            "best_epoch": best_epoch,
            "best_train_epoch": best_train_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def _validate_checkpoint(checkpoint: dict[str, Any], config: TrainConfig, input_channels: int) -> None:
    checkpoint_input_channels = int(checkpoint.get("input_channels", -1))
    if checkpoint_input_channels != input_channels:
        raise RuntimeError(
            "Checkpoint input channel count does not match the current dataset/model shape. "
            f"checkpoint={checkpoint_input_channels}, current={input_channels}"
        )
    checkpoint_model_name = str(checkpoint.get("model_name") or _checkpoint_config_value(checkpoint, "model_name") or "")
    if checkpoint_model_name and checkpoint_model_name != config.model_name:
        raise RuntimeError(
            "Checkpoint model_name does not match the current configuration. "
            f"checkpoint={checkpoint_model_name!r}, current={config.model_name!r}"
        )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    epoch_index: int,
    total_epochs: int,
    phase: str,
    log_interval: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = device.type == "cuda"
    total_loss = 0.0
    total_abs_error = 0.0
    total_values = 0
    seen_samples = 0
    started_at = time.perf_counter()
    batch_count = len(loader)

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=use_amp)
        targets = targets.to(device=device, dtype=torch.float32, non_blocking=use_amp)
        if use_amp:
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            predictions = model(inputs)
            loss = criterion(predictions, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * len(inputs)
        total_abs_error += float(torch.abs(predictions - targets).sum().item())
        total_values += int(targets.numel())
        seen_samples += len(inputs)

        should_log = batch_count > 0 and (
            batch_index == 1
            or batch_index == batch_count
            or (log_interval > 0 and batch_index % log_interval == 0)
        )
        if should_log:
            elapsed = time.perf_counter() - started_at
            avg_batch_time = elapsed / max(batch_index, 1)
            eta = avg_batch_time * (batch_count - batch_index)
            print(
                f"[{phase}] epoch {epoch_index:03d}/{total_epochs:03d} "
                f"batch {batch_index:04d}/{batch_count:04d} "
                f"loss={total_loss / max(seen_samples, 1):.6f} "
                f"mae={total_abs_error / max(total_values, 1):.6f} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
            )

    sample_count = max(len(loader.dataset), 1)
    return {
        "loss": total_loss / sample_count,
        "mae_norm": total_abs_error / max(total_values, 1),
        "elapsed_sec": time.perf_counter() - started_at,
    }


def _build_model(config: TrainConfig, in_channels: int, device: torch.device) -> nn.Module:
    return build_model(
        config.model_name,
        in_channels=in_channels,
        output_dim=24,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)


def train_from_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> None:
    config = _load_config(Path(config_path), overrides=overrides)
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    _configure_torch_runtime(device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader = _build_loader(
        config.cache_dir / f"{config.train_split_name}.npz",
        config,
        shuffle=True,
        limit_windows=config.limit_train_windows,
    )
    val_dataset, val_loader = _build_loader(
        config.cache_dir / f"{config.val_split_name}.npz",
        config,
        shuffle=False,
        limit_windows=config.limit_val_windows,
    )
    test_dataset, test_loader = _build_loader(
        config.cache_dir / f"{config.test_split_name}.npz",
        config,
        shuffle=False,
        num_workers=0,
        limit_windows=config.limit_test_windows,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty. Run prepare_dataset.py first or reduce window_size.")

    sample_input, _ = train_dataset[0]
    in_channels = int(sample_input.shape[0])
    model = _build_model(config, in_channels, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_path = config.output_dir / "best_model.pt"
    train_best_path = config.output_dir / "train_best_model.pt"
    last_path = config.output_dir / "last_checkpoint.pt"
    history_path = config.output_dir / "history.json"
    history: list[dict[str, float | int | bool]] = []
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_train_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    if config.init_checkpoint_path is not None:
        checkpoint = _load_trusted_checkpoint(config.init_checkpoint_path, device)
        _validate_checkpoint(checkpoint, config, in_channels)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized model weights from {config.init_checkpoint_path}.")

    resume_path = config.resume_checkpoint_path
    if resume_path is None and config.resume_from_last_checkpoint and last_path.exists():
        resume_path = last_path
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = _load_trusted_checkpoint(resume_path, device)
        _validate_checkpoint(checkpoint, config, in_channels)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_train_loss = float(checkpoint.get("best_train_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_train_epoch = int(checkpoint.get("best_train_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        if history_path.exists():
            try:
                existing_history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])
                if isinstance(existing_history, list):
                    history = list(existing_history)
            except json.JSONDecodeError:
                history = []
        print(f"Resumed training from {resume_path} at epoch {start_epoch:03d}.")

    _save_json(
        config.output_dir / "run_summary.json",
        {
            "device": str(device),
            "use_amp": device.type == "cuda",
            "model_name": config.model_name,
            "window_size": config.window_size,
            "window_stride": config.window_stride,
            "feature_mode": config.feature_mode,
            "require_full_window_mask": config.require_full_window_mask,
            "fill_mode": config.fill_mode,
            "max_gap": config.max_gap,
            "train_split_name": config.train_split_name,
            "val_split_name": config.val_split_name,
            "test_split_name": config.test_split_name,
            "limit_train_windows": config.limit_train_windows,
            "limit_val_windows": config.limit_val_windows,
            "limit_test_windows": config.limit_test_windows,
            "resume_from_last_checkpoint": config.resume_from_last_checkpoint,
            "resume_checkpoint_path": str(config.resume_checkpoint_path) if config.resume_checkpoint_path else None,
            "init_checkpoint_path": str(config.init_checkpoint_path) if config.init_checkpoint_path else None,
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "test_windows": len(test_dataset),
            "input_channels": in_channels,
            "subcarriers": int(sample_input.shape[1]),
            "log_interval": config.log_interval,
            "start_epoch": start_epoch,
            "best_epoch": best_epoch,
            "best_train_epoch": best_train_epoch,
            "best_val_loss": best_val_loss,
            "best_train_loss": best_train_loss,
        },
    )

    training_started_at = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            epoch_index=epoch,
            total_epochs=config.epochs,
            phase="train",
            log_interval=config.log_interval,
        )
        val_metrics = (
            _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                epoch_index=epoch,
                total_epochs=config.epochs,
                phase="val",
                log_interval=config.log_interval,
            )
            if len(val_dataset)
            else train_metrics
        )

        best_train_model_updated = False
        if best_train_loss - float(train_metrics["loss"]) > 0.0:
            best_train_loss = float(train_metrics["loss"])
            best_train_epoch = epoch
            best_train_model_updated = True
            _save_training_checkpoint(
                train_best_path,
                model,
                optimizer,
                config,
                in_channels,
                epoch,
                best_val_loss,
                best_train_loss,
                best_epoch,
                best_train_epoch,
                epochs_without_improvement,
            )

        best_model_updated = False
        if best_val_loss - float(val_metrics["loss"]) > config.early_stopping_min_delta:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            best_model_updated = True
            epochs_without_improvement = 0
            _save_training_checkpoint(
                best_path,
                model,
                optimizer,
                config,
                in_channels,
                epoch,
                best_val_loss,
                best_train_loss,
                best_epoch,
                best_train_epoch,
                epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        epoch_record: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae_norm": train_metrics["mae_norm"],
            "val_loss": val_metrics["loss"],
            "val_mae_norm": val_metrics["mae_norm"],
            "train_elapsed_sec": train_metrics["elapsed_sec"],
            "val_elapsed_sec": val_metrics["elapsed_sec"],
            "train_best_model_updated": best_train_model_updated,
            "best_model_updated": best_model_updated,
            "best_val_loss": best_val_loss,
            "best_train_loss": best_train_loss,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(epoch_record)
        _save_json(history_path, {"epochs": history})
        _save_training_checkpoint(
            last_path,
            model,
            optimizer,
            config,
            in_channels,
            epoch,
            best_val_loss,
            best_train_loss,
            best_epoch,
            best_train_epoch,
            epochs_without_improvement,
        )

        total_elapsed = time.perf_counter() - training_started_at
        print(
            f"epoch {epoch:03d}/{config.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.6f} train_mae={train_metrics['mae_norm']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_mae={val_metrics['mae_norm']:.6f} "
            f"train_best_model.pt updated={'yes' if best_train_model_updated else 'no'} "
            f"best_model.pt updated={'yes' if best_model_updated else 'no'} "
            f"patience_left={max(config.early_stopping_patience - epochs_without_improvement, 0)} "
            f"total_elapsed={_format_duration(total_elapsed)}"
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"early stopping triggered after {epochs_without_improvement} epochs without improvement.")
            break

    checkpoint_to_evaluate = best_path if best_path.exists() else last_path
    if checkpoint_to_evaluate.exists():
        _evaluate_checkpoint(model, checkpoint_to_evaluate, test_dataset, test_loader, criterion, device, config)


def _evaluate_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    test_dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float | None]:
    checkpoint = _load_trusted_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = (
        _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            epoch_index=config.epochs,
            total_epochs=config.epochs,
            phase="test",
            log_interval=config.log_interval,
        )
        if len(test_dataset)
        else {"loss": None, "mae_norm": None, "elapsed_sec": None}
    )
    _save_json(config.output_dir / "test_metrics.json", test_metrics)
    print(
        f"test_loss={test_metrics['loss']} "
        f"test_mae_norm={test_metrics['mae_norm']} "
        f"test_elapsed={_format_duration(test_metrics['elapsed_sec']) if test_metrics['elapsed_sec'] is not None else 'N/A'}"
    )
    return test_metrics


def evaluate_from_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> None:
    config = _load_config(Path(config_path), overrides=overrides)
    device = _resolve_device(config.device)
    _configure_torch_runtime(device)
    checkpoint_path = config.output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_dataset, test_loader = _build_loader(
        config.cache_dir / f"{config.test_split_name}.npz",
        config,
        shuffle=False,
        num_workers=0,
        limit_windows=config.limit_test_windows,
    )
    checkpoint = _load_trusted_checkpoint(checkpoint_path, device)
    input_channels = int(
        checkpoint.get("input_channels")
        if checkpoint.get("input_channels") is not None
        else (test_dataset[0][0].shape[0] if len(test_dataset) else 0)
    )
    if len(test_dataset) and input_channels != int(test_dataset[0][0].shape[0]):
        raise RuntimeError(
            "Checkpoint input channel count does not match the requested dataset feature mode. "
            f"checkpoint={input_channels}, dataset={int(test_dataset[0][0].shape[0])}"
        )
    if input_channels <= 0:
        raise RuntimeError("Unable to infer input channels for evaluation.")

    model_name = str(checkpoint.get("model_name") or _checkpoint_config_value(checkpoint, "model_name") or config.model_name)
    if model_name != "multi_esp_fast_pose_cnn":
        raise ValueError(f"ML2 only supports multi_esp_fast_pose_cnn checkpoints, received {model_name!r}.")
    model = _build_model(config, input_channels, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _evaluate_checkpoint(model, checkpoint_path, test_dataset, test_loader, nn.SmoothL1Loss(), device, config)
