from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_path() -> None:
    src_dir = _project_dir() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Fast CNN checkpoint to FPGA-oriented INT8 artifacts.")
    parser.add_argument("--config", type=Path, default=_project_dir() / "configs" / "fast_cnn_int8.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-cache", type=Path, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--calibration-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    payload = _load_json(config_path)
    config_dir = config_path.parent
    return {
        "config_path": config_path,
        "checkpoint": _resolve(config_dir, args.checkpoint) if args.checkpoint else _resolve(config_dir, payload.get("checkpoint")),
        "data_cache": _resolve(config_dir, args.data_cache) if args.data_cache else _resolve(config_dir, payload.get("data_cache")),
        "split": args.split or str(payload.get("split", "test")),
        "window_size": int(payload.get("window_size", 10)),
        "window_stride": int(payload.get("window_stride", 1)),
        "output_dir": _resolve(config_dir, args.output_dir) if args.output_dir else _resolve(config_dir, payload.get("output_dir")),
        "batch_size": int(args.batch_size or payload.get("batch_size", 32)),
        "calibration_windows": int(args.calibration_windows or payload.get("calibration_windows", 512)),
        "activation_percentile": float(payload.get("activation_percentile", 99.999)),
        "max_observer_samples": int(payload.get("max_observer_samples", 262144)),
        "per_update_samples": int(payload.get("per_update_samples", 16384)),
        "device": str(args.device or payload.get("device", "cpu")),
        "sample_strategy": str(payload.get("sample_strategy", "even")),
    }


def _make_loader(dataset: Any, limit_windows: int, batch_size: int, sample_strategy: str) -> DataLoader:
    limit = min(max(int(limit_windows), 0), len(dataset))
    if limit and len(dataset) > limit:
        if sample_strategy == "even":
            if limit == 1:
                indices = [len(dataset) // 2]
            else:
                step = (len(dataset) - 1) / float(limit - 1)
                indices = [min(int(round(i * step)), len(dataset) - 1) for i in range(limit)]
        elif sample_strategy == "first":
            indices = list(range(limit))
        else:
            raise ValueError(f"Unsupported sample_strategy={sample_strategy!r}")
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


def main() -> int:
    _ensure_src_path()
    from data import CachedWindowDataset
    from evaluation import checkpoint_data_settings
    from int8_export import calibrate_fast_model, export_fast_cnn_int8
    from original_models import build_model_from_checkpoint

    args = build_parser().parse_args()
    settings = _settings(args)
    checkpoint_path = settings["checkpoint"]
    data_cache = settings["data_cache"]
    output_dir = settings["output_dir"]
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if data_cache is None or not data_cache.exists():
        raise FileNotFoundError(f"Data cache not found: {data_cache}")
    if output_dir is None:
        raise RuntimeError("output_dir is required.")

    model, checkpoint = build_model_from_checkpoint(str(checkpoint_path))
    data_settings = checkpoint_data_settings(checkpoint)
    data_settings["window_size"] = settings["window_size"]
    data_settings["window_stride"] = settings["window_stride"]
    dataset = CachedWindowDataset(data_cache / f"{settings['split']}.npz", **data_settings)
    loader = _make_loader(dataset, settings["calibration_windows"], settings["batch_size"], settings["sample_strategy"])

    print(f"checkpoint={checkpoint_path}")
    print(f"data_cache={data_cache}")
    print(f"split={settings['split']} dataset_windows={len(dataset)} calibration_windows={settings['calibration_windows']}")
    calibration = calibrate_fast_model(
        model,
        loader,
        max_windows=settings["calibration_windows"],
        activation_percentile=settings["activation_percentile"],
        max_observer_samples=settings["max_observer_samples"],
        per_update_samples=settings["per_update_samples"],
        device=torch.device(settings["device"]),
    )
    metadata = export_fast_cnn_int8(
        model.cpu(),
        output_dir,
        activation_scales=calibration.activation_scales,
        calibration_report={
            "config": str(settings["config_path"]),
            "source_checkpoint": str(checkpoint_path),
            "data_cache": str(data_cache),
            "split": settings["split"],
            "windows_seen": calibration.windows_seen,
            "activation_percentile": settings["activation_percentile"],
            "observers": calibration.observer_summary,
        },
    )
    print(f"saved={output_dir}")
    print(f"layers={','.join(metadata['layers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
