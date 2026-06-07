from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_src_path(root: Path) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def remove_dir_if_requested(path: Path, *, fresh: bool) -> None:
    if fresh and path.exists():
        shutil.rmtree(path)


def run_command(command: list[str], *, cwd: Path) -> None:
    print("\n=== " + " ".join(command) + " ===", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_prepare(root: Path, args: argparse.Namespace) -> None:
    ensure_src_path(root)
    from prepare import prepare_dataset

    input_dir = args.input_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    remove_dir_if_requested(cache_dir, fresh=args.fresh)
    print(f"\n=== Preparing dataset: {input_dir} -> {cache_dir} ===", flush=True)
    prepare_dataset(
        input_dir=input_dir,
        output_dir=cache_dir,
        pattern=args.pattern,
        node_count=args.node_count,
        pair_count=args.pair_count,
        subcarrier_remap=args.subcarrier_remap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        test_files_per_group=args.test_files_per_group,
    )


def run_training(root: Path, args: argparse.Namespace) -> None:
    ensure_src_path(root)
    from training import train_from_config

    config_path = args.config.resolve()
    config_payload = load_json(config_path)
    output_dir = resolve_path(config_path.parent, config_payload["output_dir"])
    remove_dir_if_requested(output_dir, fresh=args.fresh)

    overrides = {
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir is not None else None,
        "window_stride": args.window_stride,
        "device": args.device,
    }
    print(f"\n=== Training Fast CNN: {config_path} ===", flush=True)
    train_from_config(config_path, overrides=overrides)
    print(f"\n=== Fast CNN output: {output_dir} ===", flush=True)


def build_parser() -> argparse.ArgumentParser:
    root = project_dir()
    parser = argparse.ArgumentParser(description="Prepare data, train the Fast CNN model, and optionally open the viewer.")
    parser.add_argument("--config", type=Path, default=root / "configs" / "multi_esp_fast_pose_cnn.json")
    parser.add_argument("--input-dir", type=Path, default=root.parent / "captures")
    parser.add_argument("--pattern", type=str, default="sync_csi_pose*.csv")
    parser.add_argument("--cache-dir", type=Path, default=root / "data_cache")
    parser.add_argument("--node-count", type=int, default=3)
    parser.add_argument("--pair-count", type=int, default=0)
    parser.add_argument("--subcarrier-remap", type=str, default="esp32_htltf_ht40_above_nonstbc")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-files-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fresh", action="store_true", help="Delete configured cache/output dirs before running.")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse existing cache-dir.")
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing Fast CNN checkpoint.")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False, help="Open Fast CNN viewer after training.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = project_dir()

    if not args.skip_prepare:
        run_prepare(root, args)
    else:
        print("\n=== Skipping dataset preparation ===", flush=True)

    if not args.skip_train:
        run_training(root, args)
    else:
        print("\n=== Skipping Fast CNN training ===", flush=True)

    if args.gui:
        run_command(
            [
                sys.executable,
                str(root / "scripts" / "view_gui.py"),
                "--config",
                str(args.config.resolve()),
            ],
            cwd=root,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
