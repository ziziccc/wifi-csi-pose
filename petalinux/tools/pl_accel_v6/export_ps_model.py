from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def tool_root() -> Path:
    return Path(__file__).resolve().parent


def default_ml_root() -> Path:
    return project_root() / "ML"


def ensure_ml_paths(ml_root: Path) -> None:
    src = ml_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def write_array(handle, array: np.ndarray) -> None:
    handle.write(np.ascontiguousarray(array).tobytes(order="C"))


def ensure_fast_int8_export(ml_root: Path, config_path: Path) -> Path:
    config = load_json(config_path)
    output_dir = resolve(config_path.parent, config["output_dir"])
    weights_path = output_dir / "weights_int8.npz"
    model_json_path = output_dir / "model_int8.json"
    if weights_path.exists() and model_json_path.exists():
        return output_dir

    command = [
        sys.executable,
        str(ml_root / "scripts" / "export_int8_fast_cnn.py"),
        "--config",
        str(config_path),
    ]
    subprocess.run(command, cwd=ml_root, check=True)
    if not weights_path.exists() or not model_json_path.exists():
        raise FileNotFoundError(f"INT8 artifacts were not created in {output_dir}")
    return output_dir


def write_ps_model(
    *,
    ml_root: Path,
    checkpoint_path: Path,
    int8_weights_path: Path,
    model_json_path: Path,
    output_path: Path,
) -> None:
    ensure_ml_paths(ml_root)
    import torch

    from int8_export import fold_conv_bn
    from original_models import build_model_from_checkpoint

    model, checkpoint = build_model_from_checkpoint(str(checkpoint_path))
    config = checkpoint.get("config", {}) or {}
    model_name = str(checkpoint.get("model_name") or config.get("model_name"))
    if model_name != "multi_esp_fast_pose_cnn":
        raise ValueError("Only multi_esp_fast_pose_cnn is supported by v6 ps_pose_infer.c")

    q = dict(np.load(int8_weights_path))
    metadata = load_json(model_json_path)
    if metadata.get("model_name") != "multi_esp_fast_pose_cnn":
        raise ValueError(f"Unexpected INT8 model metadata: {metadata.get('model_name')!r}")

    encoder = model.encoder.layers
    float_arrays: list[np.ndarray] = []
    for conv, bn in [(encoder[0], encoder[1]), (encoder[3], encoder[4])]:
        weight, bias = fold_conv_bn(conv, bn)
        float_arrays.append(weight.detach().cpu().numpy().astype(np.float32))
        float_arrays.append(bias.detach().cpu().numpy().astype(np.float32))

    for layer in [model.head[1], model.head[4], model.head[7]]:
        weight = layer.weight.detach().cpu().numpy().astype(np.float32)
        bias_tensor = layer.bias.detach().cpu() if layer.bias is not None else torch.zeros(layer.weight.shape[0])
        float_arrays.append(weight)
        float_arrays.append(bias_tensor.numpy().astype(np.float32))

    layers = metadata["layers"]
    scales = metadata["activation_scales"]
    pool = metadata["pool"]
    input_scale = float(scales["encoder.input"])
    output_scale = float(scales["head.fc3_out"])
    node_count = int(metadata["node_count"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(b"PSFST2\0\0")
        handle.write(
            struct.pack(
                "<IIffii",
                node_count,
                0,
                input_scale,
                output_scale,
                int(pool["requant_multiplier"]),
                int(pool["requant_shift"]),
            )
        )

        for array in float_arrays:
            write_array(handle, array)

        for key in ["conv1", "conv2", "fc1", "fc2", "fc3"]:
            write_array(handle, q[f"{key}.weight_int8"].astype(np.int8))
            write_array(handle, q[f"{key}.bias_int32"].astype(np.int32))
            write_array(handle, np.asarray(layers[key]["requant_multiplier"], dtype=np.int32))
            write_array(handle, np.asarray(layers[key]["requant_shift"], dtype=np.int32))

        for key in ["encoder.gelu1", "encoder.gelu2", "head.gelu1", "head.gelu2"]:
            array_key = metadata["luts"][key]["array"]
            write_array(handle, q[array_key].astype(np.int8))


def write_ps_inputs(
    *,
    ml_root: Path,
    config_path: Path,
    cache_dir: Path,
    windows: int,
    output_path: Path,
) -> int:
    ensure_ml_paths(ml_root)
    import torch

    from data import CachedWindowDataset
    from evaluation import checkpoint_data_settings

    config = load_json(config_path)
    checkpoint_path = resolve(config_path.parent, config["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    settings = checkpoint_data_settings(checkpoint)
    settings["window_size"] = int(config.get("window_size", settings["window_size"]))
    settings["window_stride"] = int(config.get("window_stride", settings["window_stride"]))
    dataset = CachedWindowDataset(cache_dir / "test.npz", **settings)
    count = len(dataset) if windows <= 0 else min(int(windows), len(dataset))
    raw = np.load(cache_dir / "test.npz")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(b"PSIN1\0\0\0")
        handle.write(struct.pack("<II", count, 0))
        for index in range(count):
            x, _target = dataset[index]
            raw_index = dataset.indices[index]
            file_id = int(raw["file_ids"][raw_index])
            handle.write(struct.pack("<i", file_id))
            write_array(handle, x.numpy().astype(np.float32))
    return count


def build_parser() -> argparse.ArgumentParser:
    ml_root = default_ml_root()
    this_tool = tool_root()
    parser = argparse.ArgumentParser(description="Export v6 Fast CNN PS C inference model/input binaries.")
    parser.add_argument("--ml-root", type=Path, default=ml_root)
    parser.add_argument("--config", type=Path, default=ml_root / "configs" / "fast_cnn_int8.json")
    parser.add_argument("--cache-dir", type=Path, default=this_tool / "_csv_compare_cache")
    parser.add_argument("--output-dir", type=Path, default=this_tool / "ps_export")
    parser.add_argument("--windows", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ml_root = args.ml_root.resolve()
    config_path = args.config.resolve()
    config = load_json(config_path)
    checkpoint_path = resolve(config_path.parent, config["checkpoint"])
    output_dir = ensure_fast_int8_export(ml_root, config_path)

    count = write_ps_inputs(
        ml_root=ml_root,
        config_path=config_path,
        cache_dir=args.cache_dir.resolve(),
        windows=args.windows,
        output_path=args.output_dir.resolve() / "ps_input.bin",
    )
    write_ps_model(
        ml_root=ml_root,
        checkpoint_path=checkpoint_path,
        int8_weights_path=output_dir / "weights_int8.npz",
        model_json_path=output_dir / "model_int8.json",
        output_path=args.output_dir.resolve() / "ps_model.bin",
    )
    print(json.dumps({"windows": count, "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
