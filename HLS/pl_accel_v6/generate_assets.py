from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "ML"
OUT_DIR = Path(__file__).resolve().parent
INT8_DIR = ML_DIR / "outputs" / "int8_fast_cnn"
WEIGHTS_NPZ = INT8_DIR / "weights_int8.npz"
MODEL_JSON = INT8_DIR / "model_int8.json"
PL_WEIGHT_BIN = OUT_DIR / "pl_accel_v6_weights.bin"
TOOL_WEIGHT_BIN = ROOT / "petalinux" / "tools" / "pl_accel_v6" / "pl_accel_v6_weights.bin"
WEIGHT_LAYOUT_JSON = OUT_DIR / "pl_accel_v6_weight_layout.json"

WEIGHT_MAGIC = 0x36574C50
WEIGHT_VERSION = 2

WEIGHT_OFFSETS = {
    "magic": 0,
    "version": 1,
    "total_words": 2,
    "input_scale_bits": 3,
    "output_scale_bits": 4,
    "pool_requant_mult": 5,
    "pool_requant_shift": 6,
    "reserved": 7,
    "conv1.weight_int8": 8,
    "conv1.bias_int32": 188,
    "conv1.requant_mult": 204,
    "conv1.requant_shift": 220,
    "conv2.weight_int8": 236,
    "conv2.bias_int32": 1388,
    "conv2.requant_mult": 1420,
    "conv2.requant_shift": 1452,
    "fc1.weight_int8": 1484,
    "fc1.bias_int32": 99788,
    "fc1.requant_mult": 99916,
    "fc1.requant_shift": 100044,
    "fc2.weight_int8": 100172,
    "fc2.bias_int32": 104268,
    "fc2.requant_mult": 104396,
    "fc2.requant_shift": 104524,
    "fc3.weight_int8": 104652,
    "fc3.bias_int32": 105420,
    "fc3.requant_mult": 105444,
    "fc3.requant_shift": 105468,
    "lut.encoder.gelu1.int8": 105492,
    "lut.encoder.gelu2.int8": 105556,
    "lut.head.gelu1.int8": 105620,
    "lut.head.gelu2.int8": 105684,
}
WEIGHT_WORDS = 105748

sys.path.insert(0, str(ML_DIR))
from src.int8_reference import fast_cnn_int8_forward  # noqa: E402


def c_name(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def c_type(dtype: np.dtype) -> str:
    if dtype == np.dtype("int8"):
        return "const signed char"
    if dtype == np.dtype("int32"):
        return "const int"
    if dtype == np.dtype("uint32"):
        return "const unsigned int"
    if dtype == np.dtype("float32"):
        return "const float"
    raise ValueError(f"unsupported dtype: {dtype}")


def emit_array(name: str, array: np.ndarray) -> str:
    arr = np.asarray(array)
    dims = "".join(f"[{d}]" for d in arr.shape)
    if arr.dtype == np.dtype("float32"):
        values = ", ".join(f"{float(x):.9g}f" for x in arr.reshape(-1))
    else:
        values = ", ".join(str(int(x)) for x in arr.reshape(-1))
    return f"{c_type(arr.dtype)} {c_name(name)}{dims} = {{ {values} }};\n\n"


def first_valid_window() -> np.ndarray:
    cache = ML_DIR / "outputs" / "selected_csv_cache" / "test.npz"
    if not cache.exists():
        cache = ML_DIR / "data_cache" / "test.npz"
    data = np.load(cache)
    features = data["features"].astype(np.float32)
    file_ids = data["file_ids"].astype(np.int32)
    window_size = 10
    end_index = next(
        i
        for i in range(window_size - 1, len(features))
        if (i - (window_size - 1)) % 10 == 0
        and np.all(file_ids[i - window_size + 1 : i + 1] == file_ids[i])
    )
    window = features[end_index - window_size + 1 : end_index + 1]
    return np.transpose(window, (1, 2, 0)).reshape(1, 9, 128, 10)


def quantize_to_int8(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(x.astype(np.float64) / float(scale)), -127, 127).astype(np.int8)


def float_to_u32(value: float) -> np.uint32:
    return np.uint32(struct.unpack("<I", struct.pack("<f", float(value)))[0])


def pack_int8x4_le(array: np.ndarray) -> np.ndarray:
    flat = np.asarray(array, dtype=np.int8).reshape(-1)
    if flat.size % 4 != 0:
        pad = 4 - (flat.size % 4)
        flat = np.pad(flat, (0, pad), constant_values=0).astype(np.int8)
    u8 = flat.view(np.uint8).reshape(-1, 4).astype(np.uint32)
    packed = u8[:, 0] | (u8[:, 1] << 8) | (u8[:, 2] << 16) | (u8[:, 3] << 24)
    return packed.astype("<u4")


def int32_words(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype="<i4").reshape(-1).view("<u4")


def write_words_at(blob: np.ndarray, offset: int, words: np.ndarray, name: str) -> None:
    end = offset + len(words)
    if end > len(blob):
        raise ValueError(f"{name} exceeds weight blob: {end} > {len(blob)}")
    blob[offset:end] = np.asarray(words, dtype="<u4")


def build_pl_weight_words(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> np.ndarray:
    layers = metadata["layers"]
    scales = metadata["activation_scales"]
    pool = metadata["pool"]
    blob = np.zeros(WEIGHT_WORDS, dtype="<u4")

    blob[WEIGHT_OFFSETS["magic"]] = np.uint32(WEIGHT_MAGIC)
    blob[WEIGHT_OFFSETS["version"]] = np.uint32(WEIGHT_VERSION)
    blob[WEIGHT_OFFSETS["total_words"]] = np.uint32(WEIGHT_WORDS)
    blob[WEIGHT_OFFSETS["input_scale_bits"]] = float_to_u32(scales["encoder.input"])
    blob[WEIGHT_OFFSETS["output_scale_bits"]] = float_to_u32(scales["head.fc3_out"])
    blob[WEIGHT_OFFSETS["pool_requant_mult"]] = np.uint32(int(pool["requant_multiplier"]))
    blob[WEIGHT_OFFSETS["pool_requant_shift"]] = np.uint32(int(pool["requant_shift"]))

    for layer_name in ["conv1", "conv2", "fc1", "fc2", "fc3"]:
        write_words_at(
            blob,
            WEIGHT_OFFSETS[f"{layer_name}.weight_int8"],
            pack_int8x4_le(arrays[f"{layer_name}.weight_int8"]),
            f"{layer_name}.weight_int8",
        )
        write_words_at(
            blob,
            WEIGHT_OFFSETS[f"{layer_name}.bias_int32"],
            int32_words(arrays[f"{layer_name}.bias_int32"]),
            f"{layer_name}.bias_int32",
        )
        write_words_at(
            blob,
            WEIGHT_OFFSETS[f"{layer_name}.requant_mult"],
            int32_words(np.asarray(layers[layer_name]["requant_multiplier"], dtype=np.int32)),
            f"{layer_name}.requant_mult",
        )
        write_words_at(
            blob,
            WEIGHT_OFFSETS[f"{layer_name}.requant_shift"],
            int32_words(np.asarray(layers[layer_name]["requant_shift"], dtype=np.int32)),
            f"{layer_name}.requant_shift",
        )

    lut_keys = [
        ("lut.encoder.gelu1.int8", "lut.encoder.gelu1.int8"),
        ("lut.encoder.gelu2.int8", "lut.encoder.gelu2.int8"),
        ("lut.head.gelu1.int8", "lut.head.gelu1.int8"),
        ("lut.head.gelu2.int8", "lut.head.gelu2.int8"),
    ]
    for offset_key, array_key in lut_keys:
        write_words_at(blob, WEIGHT_OFFSETS[offset_key], pack_int8x4_le(arrays[array_key]), array_key)

    return blob


def write_pl_weight_bin(arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> np.ndarray:
    blob = build_pl_weight_words(arrays, metadata)
    payload = blob.astype("<u4").tobytes()
    PL_WEIGHT_BIN.write_bytes(payload)
    if TOOL_WEIGHT_BIN.parent.exists():
        TOOL_WEIGHT_BIN.write_bytes(payload)
    layout = {
        "magic": f"0x{WEIGHT_MAGIC:08x}",
        "version": WEIGHT_VERSION,
        "word_bytes": 4,
        "total_words": WEIGHT_WORDS,
        "total_bytes": WEIGHT_WORDS * 4,
        "offset_words": WEIGHT_OFFSETS,
    }
    WEIGHT_LAYOUT_JSON.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    return blob


def write_test_vectors(arrays: dict[str, np.ndarray], metadata: dict[str, object], weight_blob: np.ndarray) -> None:
    x_float = first_valid_window()
    input_scale = float(metadata["activation_scales"]["encoder.input"])
    x_q = quantize_to_int8(x_float, input_scale)
    expected, _ = fast_cnn_int8_forward(x_float, arrays, metadata)

    text = ["#pragma once\n\n"]
    text.append(emit_array("test.input", x_q.reshape(-1)))
    text.append(emit_array("test.weights", weight_blob.astype(np.uint32)))
    text.append(emit_array("test.expected_pose", expected.reshape(-1).astype(np.float32)))
    (OUT_DIR / "test_vectors.h").write_text("".join(text), encoding="utf-8")


def main() -> int:
    if not WEIGHTS_NPZ.exists():
        raise FileNotFoundError(WEIGHTS_NPZ)
    if not MODEL_JSON.exists():
        raise FileNotFoundError(MODEL_JSON)

    arrays = dict(np.load(WEIGHTS_NPZ))
    metadata = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    weight_blob = write_pl_weight_bin(arrays, metadata)
    write_test_vectors(arrays, metadata, weight_blob)
    print(f"wrote {PL_WEIGHT_BIN}")
    if TOOL_WEIGHT_BIN.exists():
        print(f"wrote {TOOL_WEIGHT_BIN}")
    print(f"wrote {WEIGHT_LAYOUT_JSON}")
    print(f"wrote {OUT_DIR / 'test_vectors.h'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
