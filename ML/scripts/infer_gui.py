from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_path() -> None:
    src_dir = project_dir() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def run_command(command: list[str], *, cwd: Path) -> None:
    print("\n=== " + " ".join(command) + " ===", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def build_parser() -> argparse.ArgumentParser:
    root = project_dir()
    parser = argparse.ArgumentParser(description="Prepare selected CSV files and open the Fast CNN pose viewer.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=root / "infer_csv",
        help="CSV file or directory containing synchronized CSI+pose CSV files.",
    )
    parser.add_argument("--pattern", type=str, default="sync_csi_pose*.csv", help="Used only when input is a directory.")
    parser.add_argument("--config", type=Path, default=root / "configs" / "multi_esp_fast_pose_cnn.json")
    parser.add_argument("--cache-dir", type=Path, default=root / "outputs" / "selected_csv_cache")
    parser.add_argument("--windows", type=int, default=0, help="0 means every valid window.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--node-count", type=int, default=3)
    parser.add_argument("--pair-count", type=int, default=0)
    parser.add_argument("--subcarrier-remap", type=str, default="esp32_htltf_ht40_above_nonstbc")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--int8-dir", type=Path, default=root / "outputs" / "int8_fast_cnn")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 reference overlay in the viewer.")
    parser.add_argument("--log-interval", type=int, default=1, help="Print viewer prediction progress every N batches.")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse existing cache-dir/test.npz.")
    parser.add_argument("--no-gui", action="store_true", help="Only prepare the cache; do not open the viewer.")
    return parser


def _prepare_test_cache(
    *,
    input_dir: Path,
    pattern: str,
    cache_dir: Path,
    node_count: int,
    pair_count: int,
    subcarrier_remap: str,
) -> dict[str, Any]:
    _ensure_src_path()
    from prepare import (
        ESP32_HT40_ABOVE_NONSTBC_LAYOUT,
        ESP32_HTLTF_HT40_ABOVE_NONSTBC_LAYOUT,
        POSE_FIELDS,
        _infer_pair_count,
        _save_split,
    )

    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} with pattern {pattern!r}.")

    raw_pair_count = int(pair_count)
    if raw_pair_count <= 0:
        raw_pair_count = int(_infer_pair_count(files))

    output_pair_count = raw_pair_count
    axis_metadata: dict[str, Any] = {
        "mode": subcarrier_remap,
        "raw_pair_count": raw_pair_count,
        "output_pair_count": output_pair_count,
    }
    if subcarrier_remap == "esp32_htltf_ht40_above_nonstbc":
        output_pair_count = len(ESP32_HTLTF_HT40_ABOVE_NONSTBC_LAYOUT["axis"])
        axis_metadata = {
            "mode": subcarrier_remap,
            "raw_pair_count": raw_pair_count,
            "output_pair_count": output_pair_count,
            "signed_axis": ESP32_HTLTF_HT40_ABOVE_NONSTBC_LAYOUT["axis"],
            "segments": [
                {
                    "name": str(ESP32_HTLTF_HT40_ABOVE_NONSTBC_LAYOUT["segment"]["name"]),
                    "raw_indices": list(ESP32_HTLTF_HT40_ABOVE_NONSTBC_LAYOUT["segment"]["raw_indices"]),
                }
            ],
        }
    elif subcarrier_remap == "esp32_ht40_above_nonstbc":
        output_pair_count = len(ESP32_HT40_ABOVE_NONSTBC_LAYOUT["axis"]) * 2
        axis_metadata = {
            "mode": subcarrier_remap,
            "raw_pair_count": raw_pair_count,
            "output_pair_count": output_pair_count,
            "signed_axis": ESP32_HT40_ABOVE_NONSTBC_LAYOUT["axis"],
            "segments": [
                {
                    "name": str(segment["name"]),
                    "raw_indices": list(segment["raw_indices"]),
                }
                for segment in ESP32_HT40_ABOVE_NONSTBC_LAYOUT["segments"]
            ],
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = _save_split(
        split_name="test",
        files=files,
        output_dir=cache_dir,
        node_count=node_count,
        pair_count=raw_pair_count,
        output_pair_count=output_pair_count,
        subcarrier_remap=subcarrier_remap,
    )
    for split_name in ("train", "val"):
        np.savez_compressed(
            cache_dir / f"{split_name}.npz",
            features=np.zeros((0, node_count * 3, output_pair_count), dtype=np.float32),
            labels=np.zeros((0, len(POSE_FIELDS)), dtype=np.float32),
            file_ids=np.zeros((0,), dtype=np.int32),
            trigger_seq=np.zeros((0,), dtype=np.int64),
            frame_size=np.zeros((0, 2), dtype=np.int32),
        )

    metadata = {
        "input_dir": str(input_dir),
        "pattern": pattern,
        "node_count": node_count,
        "pair_count": output_pair_count,
        "raw_pair_count": raw_pair_count,
        "feature_channels": node_count * 3,
        "subcarrier_remap": subcarrier_remap,
        "subcarrier_axis": axis_metadata,
        "pose_fields": POSE_FIELDS,
        "splits": {
            "train": {"files": [], "summary": {"files": 0, "frames": 0}},
            "val": {"files": [], "summary": {"files": 0, "frames": 0}},
            "test": {"files": [str(path) for path in files], "summary": summary},
        },
    }
    (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    args = build_parser().parse_args()
    root = project_dir()
    input_path = args.input.resolve()
    print(f"[infer] input={input_path}", flush=True)
    if input_path.is_file():
        input_dir = input_path.parent
        pattern = input_path.name
    else:
        input_dir = input_path
        pattern = args.pattern
    print(f"[infer] input_dir={input_dir} pattern={pattern}", flush=True)

    cache_dir = args.cache_dir.resolve()
    if not args.skip_prepare:
        print(f"[infer] preparing cache={cache_dir}", flush=True)
        metadata = _prepare_test_cache(
            input_dir=input_dir,
            pattern=pattern,
            cache_dir=cache_dir,
            node_count=args.node_count,
            pair_count=args.pair_count,
            subcarrier_remap=args.subcarrier_remap,
        )
        print(
            "prepared "
            f"files={metadata['splits']['test']['summary']['files']} "
            f"frames={metadata['splits']['test']['summary']['frames']} "
            f"cache={cache_dir}"
        )
    elif not (cache_dir / "test.npz").exists():
        raise FileNotFoundError(f"--skip-prepare was set but cache does not exist: {cache_dir / 'test.npz'}")
    else:
        print(f"[infer] reusing cache={cache_dir}", flush=True)

    if args.no_gui:
        print("[infer] --no-gui set; done after cache preparation", flush=True)
        return 0

    print("[infer] launching overlay viewer", flush=True)
    command = [
        sys.executable,
        str(root / "scripts" / "view_gui.py"),
        "--config",
        str(args.config.resolve()),
        "--cache-dir",
        str(cache_dir),
        "--split",
        "test",
        "--windows",
        str(args.windows),
        "--device",
        args.device,
        "--int8-dir",
        str(args.int8_dir.resolve()),
        "--log-interval",
        str(args.log_interval),
    ]
    if args.no_int8:
        command.append("--no-int8")
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    run_command(command, cwd=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
