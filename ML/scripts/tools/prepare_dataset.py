from __future__ import annotations

import argparse
import sys
from pathlib import Path


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_path() -> None:
    src_dir = project_dir() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ML2 Fast CNN CSI pose cache from CSV files.")
    parser.add_argument("--input-dir", type=Path, default=project_dir().parent / "captures")
    parser.add_argument("--output-dir", type=Path, default=project_dir() / "data_cache")
    parser.add_argument("--pattern", type=str, default="sync_csi_pose*.csv")
    parser.add_argument("--node-count", type=int, default=3)
    parser.add_argument("--pair-count", type=int, default=0, help="Use 0 to infer from CSV.")
    parser.add_argument("--subcarrier-remap", type=str, default="esp32_htltf_ht40_above_nonstbc")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-files-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    _ensure_src_path()
    from prepare import prepare_dataset

    args = build_parser().parse_args()
    prepare_dataset(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        pattern=args.pattern,
        node_count=args.node_count,
        pair_count=args.pair_count,
        subcarrier_remap=args.subcarrier_remap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        test_files_per_group=args.test_files_per_group,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
