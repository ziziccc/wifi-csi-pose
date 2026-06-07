from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

INT8_QMIN = -127
INT8_QMAX = 127


def load_int8_artifacts(output_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    output_dir = Path(output_dir)
    arrays = dict(np.load(output_dir / "weights_int8.npz"))
    metadata = json.loads((output_dir / "model_int8.json").read_text(encoding="utf-8"))
    return arrays, metadata


def quantize_to_int8(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(x.astype(np.float64) / float(scale)), INT8_QMIN, INT8_QMAX).astype(np.int8)


def _rounding_right_shift(values: np.ndarray, shift: int) -> np.ndarray:
    values = values.astype(np.int64, copy=False)
    if shift <= 0:
        return values << (-shift)
    offset = np.int64(1 << (shift - 1))
    rounded = np.where(values >= 0, values + offset, values - offset)
    return rounded >> shift


def requantize_acc(acc: np.ndarray, multipliers: list[int], shifts: list[int]) -> np.ndarray:
    out = np.empty(acc.shape, dtype=np.int8)
    for channel, (multiplier, shift) in enumerate(zip(multipliers, shifts)):
        values = acc[:, channel].astype(np.int64) * np.int64(multiplier)
        out[:, channel] = np.clip(_rounding_right_shift(values, int(shift)), INT8_QMIN, INT8_QMAX).astype(np.int8)
    return out


def dequantize_acc(acc: np.ndarray, input_scale: float, weight_scales: np.ndarray) -> np.ndarray:
    return acc.astype(np.float64) * float(input_scale) * weight_scales.reshape(1, -1)


def conv2d_int8(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    layer_meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    pad_h, pad_w = layer_meta["padding"]
    stride_h, stride_w = layer_meta["stride"]
    padded = np.pad(x.astype(np.int16), ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out_h = ((in_h + 2 * pad_h - kernel_h) // stride_h) + 1
    out_w = ((in_w + 2 * pad_w - kernel_w) // stride_w) + 1
    acc = np.empty((batch, out_channels, out_h, out_w), dtype=np.int32)
    weight_i32 = weight.astype(np.int32)
    for b in range(batch):
        for oc in range(out_channels):
            for oh in range(out_h):
                h0 = oh * stride_h
                for ow in range(out_w):
                    w0 = ow * stride_w
                    region = padded[b, :, h0 : h0 + kernel_h, w0 : w0 + kernel_w].astype(np.int32)
                    acc[b, oc, oh, ow] = int(np.sum(region * weight_i32[oc]) + int(bias[oc]))
    flat_acc = acc.reshape(batch, out_channels, -1)
    q = requantize_acc(flat_acc, layer_meta["requant_multiplier"], layer_meta["requant_shift"])
    return acc, q.reshape(batch, out_channels, out_h, out_w)


def linear_int8(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    layer_meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    acc = x.astype(np.int32) @ weight.astype(np.int32).T
    acc = acc + bias.astype(np.int32).reshape(1, -1)
    q = requantize_acc(acc.reshape(acc.shape[0], acc.shape[1], 1), layer_meta["requant_multiplier"], layer_meta["requant_shift"])
    return acc.astype(np.int32), q.reshape(acc.shape)


def apply_lut(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[x.astype(np.int16) + 128].astype(np.int8)


def _adaptive_bin(index: int, input_size: int, output_size: int) -> tuple[int, int]:
    start = int(np.floor(index * input_size / output_size))
    end = int(np.ceil((index + 1) * input_size / output_size))
    return start, end


def adaptive_avg_pool2d_int8(x: np.ndarray, pool_meta: dict[str, Any]) -> np.ndarray:
    batch, channels, in_h, in_w = x.shape
    out_h, out_w = pool_meta["output_size"]
    out = np.empty((batch, channels, out_h, out_w), dtype=np.int8)
    multiplier = int(pool_meta["requant_multiplier"])
    shift = int(pool_meta["requant_shift"])
    for oh in range(out_h):
        h0, h1 = _adaptive_bin(oh, in_h, out_h)
        for ow in range(out_w):
            w0, w1 = _adaptive_bin(ow, in_w, out_w)
            region = x[:, :, h0:h1, w0:w1].astype(np.int32)
            summed = region.sum(axis=(2, 3))
            divisor = max((h1 - h0) * (w1 - w0), 1)
            values = summed.astype(np.int64) * np.int64(multiplier)
            pooled = _rounding_right_shift(values, shift)
            pooled = np.where(pooled >= 0, (pooled + divisor // 2) // divisor, (pooled - divisor // 2) // divisor)
            out[:, :, oh, ow] = np.clip(pooled, INT8_QMIN, INT8_QMAX).astype(np.int8)
    return out


def fast_cnn_int8_forward(
    x_float: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    dump_intermediates: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scales = metadata["activation_scales"]
    layers = metadata["layers"]
    luts = metadata["luts"]
    node_count = int(metadata["node_count"])
    dumps: dict[str, np.ndarray] = {}
    receiver_features: list[np.ndarray] = []

    for node_index in range(node_count):
        base = x_float[:, node_index : node_index + 1]
        delta = x_float[:, node_count + node_index : node_count + node_index + 1]
        mask = x_float[:, node_count * 2 + node_index : node_count * 2 + node_index + 1]
        hidden = np.concatenate([base, delta, mask], axis=1)
        hidden = quantize_to_int8(hidden, scales["encoder.input"])
        conv1_acc, hidden = conv2d_int8(hidden, arrays["conv1.weight_int8"], arrays["conv1.bias_int32"], layers["conv1"])
        if dump_intermediates:
            dumps[f"encoder.node{node_index}.conv1_acc_int32"] = conv1_acc
            dumps[f"encoder.node{node_index}.conv1_out_int8"] = hidden
        hidden = apply_lut(hidden, arrays[luts["encoder.gelu1"]["array"]])
        if dump_intermediates:
            dumps[f"encoder.node{node_index}.gelu1_int8"] = hidden
        conv2_acc, hidden = conv2d_int8(hidden, arrays["conv2.weight_int8"], arrays["conv2.bias_int32"], layers["conv2"])
        if dump_intermediates:
            dumps[f"encoder.node{node_index}.conv2_acc_int32"] = conv2_acc
            dumps[f"encoder.node{node_index}.conv2_out_int8"] = hidden
        hidden = apply_lut(hidden, arrays[luts["encoder.gelu2"]["array"]])
        if dump_intermediates:
            dumps[f"encoder.node{node_index}.gelu2_int8"] = hidden
        hidden = adaptive_avg_pool2d_int8(hidden, metadata["pool"])
        if dump_intermediates:
            dumps[f"encoder.node{node_index}.pool_int8"] = hidden
        receiver_features.append(hidden)

    hidden = np.concatenate(receiver_features, axis=1)
    hidden = hidden.reshape(hidden.shape[0], -1).astype(np.int8)
    if dump_intermediates:
        dumps["head.flatten_int8"] = hidden
    fc1_acc, hidden = linear_int8(hidden, arrays["fc1.weight_int8"], arrays["fc1.bias_int32"], layers["fc1"])
    if dump_intermediates:
        dumps["head.fc1_acc_int32"] = fc1_acc
        dumps["head.fc1_out_int8"] = hidden
    hidden = apply_lut(hidden, arrays[luts["head.gelu1"]["array"]])
    if dump_intermediates:
        dumps["head.gelu1_int8"] = hidden
    fc2_acc, hidden = linear_int8(hidden, arrays["fc2.weight_int8"], arrays["fc2.bias_int32"], layers["fc2"])
    if dump_intermediates:
        dumps["head.fc2_acc_int32"] = fc2_acc
        dumps["head.fc2_out_int8"] = hidden
    hidden = apply_lut(hidden, arrays[luts["head.gelu2"]["array"]])
    if dump_intermediates:
        dumps["head.gelu2_int8"] = hidden
    fc3_acc, pose_int8 = linear_int8(hidden, arrays["fc3.weight_int8"], arrays["fc3.bias_int32"], layers["fc3"])
    pose_float = pose_int8.astype(np.float32) * np.float32(scales["head.fc3_out"])
    if dump_intermediates:
        dumps["head.fc3_acc_int32"] = fc3_acc
        dumps["pose_int8"] = pose_int8
        dumps["pose_float"] = pose_float
    return pose_float.astype(np.float32), dumps
