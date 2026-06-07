from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_path() -> None:
    src_dir = _project_dir() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an ML2 Fast CNN checkpoint structure.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint path. Can be provided more than once.",
    )
    return parser


def main() -> int:
    _ensure_src_path()
    from original_models import build_model_from_checkpoint

    args = build_parser().parse_args()
    for checkpoint_path in args.checkpoint:
        model, checkpoint = build_model_from_checkpoint(checkpoint_path)
        config = checkpoint.get("config", {}) or {}
        print("=" * 80)
        print(f"checkpoint: {checkpoint_path}")
        print(f"model_name: {checkpoint.get('model_name')}")
        print(f"input_channels: {checkpoint.get('input_channels')}")
        print(f"window_size: {config.get('window_size')} feature_mode: {config.get('feature_mode')}")
        print(f"hidden_dim: {config.get('hidden_dim')} dropout: {config.get('dropout')}")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"parameters: {parameter_count:,}")
        for key, tensor in checkpoint["model_state_dict"].items():
            if torch.is_tensor(tensor):
                print(f"{key:42s} {tuple(tensor.shape)} {tensor.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
