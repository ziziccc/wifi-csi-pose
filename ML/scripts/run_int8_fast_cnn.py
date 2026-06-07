from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    parser = argparse.ArgumentParser(description="Run the true-integer INT8 Fast CNN reference and compare to float32.")
    parser.add_argument("--config", type=Path, default=_project_dir() / "configs" / "fast_cnn_int8.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-cache", type=Path, default=None)
    parser.add_argument("--int8-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dump", type=Path, default=None, help="Optional .npz path for layer dumps from the first batch.")
    return parser


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    payload = _load_json(config_path)
    config_dir = config_path.parent
    return {
        "checkpoint": _resolve(config_dir, args.checkpoint) if args.checkpoint else _resolve(config_dir, payload.get("checkpoint")),
        "data_cache": _resolve(config_dir, args.data_cache) if args.data_cache else _resolve(config_dir, payload.get("data_cache")),
        "int8_dir": _resolve(config_dir, args.int8_dir) if args.int8_dir else _resolve(config_dir, payload.get("output_dir")),
        "split": args.split or str(payload.get("split", "test")),
        "window_size": int(payload.get("window_size", 10)),
        "window_stride": int(payload.get("window_stride", 1)),
        "windows": int(args.windows if args.windows is not None else payload.get("eval_windows", 64)),
        "batch_size": int(args.batch_size or payload.get("batch_size", 32)),
        "device": str(args.device or payload.get("device", "cpu")),
    }


def main() -> int:
    _ensure_src_path()
    from data import CachedWindowDataset
    from evaluation import checkpoint_data_settings
    from int8_reference import fast_cnn_int8_forward, load_int8_artifacts
    from original_models import build_model_from_checkpoint

    args = build_parser().parse_args()
    settings = _settings(args)
    model, checkpoint = build_model_from_checkpoint(str(settings["checkpoint"]))
    data_settings = checkpoint_data_settings(checkpoint)
    data_settings["window_size"] = settings["window_size"]
    data_settings["window_stride"] = settings["window_stride"]
    dataset = CachedWindowDataset(settings["data_cache"] / f"{settings['split']}.npz", **data_settings)
    limit = min(len(dataset), max(settings["windows"], 0))
    if limit <= 0:
        limit = len(dataset)
    subset = Subset(dataset, list(range(limit)))
    loader = DataLoader(subset, batch_size=settings["batch_size"], shuffle=False, num_workers=0)
    arrays, metadata = load_int8_artifacts(settings["int8_dir"])

    model.eval()
    device = torch.device(settings["device"])
    model.to(device)
    float_outputs: list[np.ndarray] = []
    int8_outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    first_dump: dict[str, np.ndarray] | None = None
    with torch.no_grad():
        for batch_index, (inputs, target) in enumerate(loader):
            float_pred = model(inputs.to(device=device, dtype=torch.float32)).cpu().numpy().astype(np.float32)
            int8_pred, dumps = fast_cnn_int8_forward(
                inputs.numpy().astype(np.float32),
                arrays,
                metadata,
                dump_intermediates=(batch_index == 0 and args.dump is not None),
            )
            if dumps and first_dump is None:
                first_dump = dumps
            float_outputs.append(float_pred)
            int8_outputs.append(int8_pred)
            targets.append(target.numpy().astype(np.float32))

    float_array = np.concatenate(float_outputs, axis=0)
    int8_array = np.concatenate(int8_outputs, axis=0)
    target_array = np.concatenate(targets, axis=0)
    diff = np.abs(int8_array - float_array)
    target_error = np.abs(int8_array - target_array)
    summary = {
        "windows": int(float_array.shape[0]),
        "int8_minus_float_mae": float(diff.mean()),
        "int8_minus_float_max_abs": float(diff.max()),
        "int8_mae_to_target": float(target_error.mean()),
    }
    print(json.dumps(summary, indent=2))
    if args.dump is not None:
        dump_path = args.dump.resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_path,
            float32_pose=float_array,
            int8_pose_float=int8_array,
            target_pose=target_array,
            **(first_dump or {}),
        )
        print(f"dump={dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
