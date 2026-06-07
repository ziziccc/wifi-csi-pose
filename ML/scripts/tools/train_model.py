from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_path() -> None:
    src_dir = _project_dir() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the ML2 Fast CNN pose model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--feature-mode", type=str, choices=["all", "base_only"], default=None)
    parser.add_argument("--require-full-window-mask", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fill-mode", type=str, choices=["zero", "forward_fill"], default=None)
    parser.add_argument("--max-gap", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from-last-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> int:
    _ensure_src_path()
    from training import evaluate_from_config, train_from_config

    args = build_parser().parse_args()
    config_payload = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    window_stride = int(args.window_stride or config_payload.get("window_stride", 1))
    overrides = {
        "feature_mode": args.feature_mode,
        "require_full_window_mask": args.require_full_window_mask,
        "fill_mode": args.fill_mode,
        "max_gap": args.max_gap,
        "output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "init_checkpoint_path": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "resume_from_last_checkpoint": args.resume_from_last_checkpoint,
        "window_stride": window_stride,
    }
    print(f"ML2 Fast CNN window_stride={window_stride}")
    if args.test_only:
        evaluate_from_config(args.config, overrides=overrides)
    else:
        train_from_config(args.config, overrides=overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
