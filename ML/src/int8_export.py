from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from original_models import MultiESPFastPoseCNN

INT8_QMIN = -127
INT8_QMAX = 127
SCALE_EPS = 1.0e-12


def safe_scale(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    return max(float(value), SCALE_EPS)


def quantize_multiplier(real_scale: float) -> tuple[int, int]:
    real_scale = float(real_scale)
    if real_scale <= 0.0:
        return 0, 0
    significand, exponent = math.frexp(real_scale)
    multiplier = int(round(significand * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1
    shift = 31 - exponent
    return multiplier, shift


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight.detach().float().cpu()
    bias = torch.zeros(weight.shape[0], dtype=torch.float32) if conv.bias is None else conv.bias.detach().float().cpu()
    gamma = bn.weight.detach().float().cpu()
    beta = bn.bias.detach().float().cpu()
    running_mean = bn.running_mean.detach().float().cpu()
    running_var = bn.running_var.detach().float().cpu()
    inv_std = torch.rsqrt(running_var + float(bn.eps))
    bn_scale = gamma * inv_std
    fused_weight = weight * bn_scale.reshape(-1, 1, 1, 1)
    fused_bias = beta + (bias - running_mean) * bn_scale
    return fused_weight, fused_bias


def quantize_weight_per_output_channel(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    weight_f = weight.detach().float().cpu()
    flat = weight_f.reshape(weight_f.shape[0], -1)
    scales = torch.clamp(flat.abs().amax(dim=1) / float(INT8_QMAX), min=SCALE_EPS)
    view_shape = (weight_f.shape[0],) + (1,) * (weight_f.ndim - 1)
    q_weight = torch.round(weight_f / scales.view(view_shape)).clamp(INT8_QMIN, INT8_QMAX).to(torch.int8)
    return q_weight.numpy(), scales.numpy().astype(np.float32)


def quantize_bias_int32(bias: torch.Tensor, input_scale: float, weight_scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bias_f = bias.detach().float().cpu().numpy()
    bias_scales = np.maximum(weight_scales.astype(np.float64) * safe_scale(input_scale), SCALE_EPS)
    q_bias = np.round(bias_f / bias_scales).clip(-(2**31), 2**31 - 1).astype(np.int32)
    return q_bias, bias_scales.astype(np.float32)


class SampledAbsObserver:
    def __init__(self, max_samples: int, per_update_samples: int) -> None:
        self.max_samples = int(max_samples)
        self.per_update_samples = int(per_update_samples)
        self.samples: torch.Tensor | None = None
        self.max_abs = 0.0
        self.update_count = 0

    def update(self, value: torch.Tensor) -> None:
        detached = value.detach().float().abs().flatten().cpu()
        if detached.numel() == 0:
            return
        self.max_abs = max(self.max_abs, float(detached.max().item()))
        self.update_count += 1
        if detached.numel() > self.per_update_samples:
            step = max(detached.numel() // self.per_update_samples, 1)
            detached = detached[::step][: self.per_update_samples]
        combined = detached if self.samples is None else torch.cat((self.samples, detached), dim=0)
        if combined.numel() > self.max_samples:
            indices = torch.linspace(0, combined.numel() - 1, self.max_samples).long()
            combined = combined[indices]
        self.samples = combined

    def scale(self, percentile: float) -> float:
        if self.samples is None or self.samples.numel() == 0:
            return 1.0
        percentile = min(max(float(percentile), 0.0), 100.0)
        if percentile >= 100.0:
            max_abs = self.max_abs
        else:
            max_abs = float(torch.quantile(self.samples, percentile / 100.0).item())
            max_abs = min(max(max_abs, SCALE_EPS), max(self.max_abs, SCALE_EPS))
        return safe_scale(max_abs / float(INT8_QMAX))


class CalibrationObservers:
    def __init__(self, max_samples: int, per_update_samples: int) -> None:
        self.max_samples = int(max_samples)
        self.per_update_samples = int(per_update_samples)
        self.observers: dict[str, SampledAbsObserver] = {}

    def observe(self, key: str, value: torch.Tensor) -> None:
        observer = self.observers.get(key)
        if observer is None:
            observer = SampledAbsObserver(self.max_samples, self.per_update_samples)
            self.observers[key] = observer
        observer.update(value)

    def scales(self, percentile: float) -> dict[str, float]:
        return {key: observer.scale(percentile) for key, observer in sorted(self.observers.items())}

    def summary(self, percentile: float) -> dict[str, dict[str, float | int]]:
        return {
            key: {
                "scale": observer.scale(percentile),
                "max_abs": observer.max_abs,
                "updates": observer.update_count,
                "sample_count": 0 if observer.samples is None else int(observer.samples.numel()),
            }
            for key, observer in sorted(self.observers.items())
        }


@dataclass
class CalibrationResult:
    activation_scales: dict[str, float]
    observer_summary: dict[str, dict[str, float | int]]
    windows_seen: int


def observe_fast_model(model: MultiESPFastPoseCNN, inputs: torch.Tensor, observers: CalibrationObservers) -> None:
    encoder_layers = model.encoder.layers
    receiver_features: list[torch.Tensor] = []
    for node_index in range(model.node_count):
        hidden = model._slice_receiver_channels(inputs, node_index)
        observers.observe("encoder.input", hidden)
        hidden = encoder_layers[0](hidden)
        hidden = encoder_layers[1](hidden)
        observers.observe("encoder.conv1_out", hidden)
        hidden = encoder_layers[2](hidden)
        observers.observe("encoder.gelu1", hidden)
        hidden = encoder_layers[3](hidden)
        hidden = encoder_layers[4](hidden)
        observers.observe("encoder.conv2_out", hidden)
        hidden = encoder_layers[5](hidden)
        observers.observe("encoder.gelu2", hidden)
        hidden = encoder_layers[6](hidden)
        observers.observe("encoder.pool", hidden)
        receiver_features.append(hidden)
    hidden = torch.cat(receiver_features, dim=1)
    hidden = model.head[0](hidden)
    observers.observe("head.flatten", hidden)
    hidden = model.head[1](hidden)
    observers.observe("head.fc1_out", hidden)
    hidden = model.head[2](hidden)
    observers.observe("head.gelu1", hidden)
    hidden = model.head[3](hidden)
    hidden = model.head[4](hidden)
    observers.observe("head.fc2_out", hidden)
    hidden = model.head[5](hidden)
    observers.observe("head.gelu2", hidden)
    hidden = model.head[6](hidden)
    hidden = model.head[7](hidden)
    observers.observe("head.fc3_out", hidden)


def calibrate_fast_model(
    model: MultiESPFastPoseCNN,
    loader: DataLoader,
    *,
    max_windows: int,
    activation_percentile: float,
    max_observer_samples: int,
    per_update_samples: int,
    device: torch.device | str,
) -> CalibrationResult:
    model.eval()
    model.to(device)
    observers = CalibrationObservers(max_observer_samples, per_update_samples)
    windows_seen = 0
    with torch.no_grad():
        for inputs, _targets in loader:
            remaining = int(max_windows) - windows_seen if max_windows > 0 else inputs.shape[0]
            if remaining <= 0:
                break
            if inputs.shape[0] > remaining:
                inputs = inputs[:remaining]
            observe_fast_model(model, inputs.to(device), observers)
            windows_seen += int(inputs.shape[0])
    return CalibrationResult(
        activation_scales=observers.scales(activation_percentile),
        observer_summary=observers.summary(activation_percentile),
        windows_seen=windows_seen,
    )


def gelu_lut(input_scale: float, output_scale: float) -> np.ndarray:
    codes = torch.arange(-128, 128, dtype=torch.float32)
    values = F.gelu(codes * safe_scale(input_scale))
    return torch.round(values / safe_scale(output_scale)).clamp(INT8_QMIN, INT8_QMAX).to(torch.int8).numpy()


def _layer_metadata(
    *,
    layer_type: str,
    weight_shape: list[int],
    input_scale: float,
    output_scale: float,
    weight_scales: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    multipliers = []
    shifts = []
    for weight_scale in weight_scales:
        multiplier, shift = quantize_multiplier(safe_scale(input_scale) * safe_scale(float(weight_scale)) / safe_scale(output_scale))
        multipliers.append(multiplier)
        shifts.append(shift)
    payload: dict[str, Any] = {
        "type": layer_type,
        "weight_shape": weight_shape,
        "input_scale": safe_scale(input_scale),
        "output_scale": safe_scale(output_scale),
        "requant_multiplier": multipliers,
        "requant_shift": shifts,
    }
    if extra:
        payload.update(extra)
    return payload


def export_fast_cnn_int8(
    model: MultiESPFastPoseCNN,
    output_dir: str | Path,
    *,
    activation_scales: dict[str, float],
    calibration_report: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    layers: dict[str, Any] = {}

    encoder = model.encoder.layers
    conv_specs = [
        ("conv1", encoder[0], encoder[1], "encoder.input", "encoder.conv1_out"),
        ("conv2", encoder[3], encoder[4], "encoder.gelu1", "encoder.conv2_out"),
    ]
    for name, conv, bn, input_key, output_key in conv_specs:
        fused_weight, fused_bias = fold_conv_bn(conv, bn)
        q_weight, weight_scales = quantize_weight_per_output_channel(fused_weight)
        q_bias, bias_scales = quantize_bias_int32(fused_bias, activation_scales[input_key], weight_scales)
        arrays[f"{name}.weight_int8"] = q_weight
        arrays[f"{name}.weight_scale"] = weight_scales
        arrays[f"{name}.bias_int32"] = q_bias
        arrays[f"{name}.bias_scale"] = bias_scales
        layers[name] = _layer_metadata(
            layer_type="conv2d",
            weight_shape=list(q_weight.shape),
            input_scale=activation_scales[input_key],
            output_scale=activation_scales[output_key],
            weight_scales=weight_scales,
            extra={
                "stride": list(conv.stride),
                "padding": list(conv.padding),
                "dilation": list(conv.dilation),
                "groups": int(conv.groups),
                "input_scale_key": input_key,
                "output_scale_key": output_key,
            },
        )

    linear_specs = [
        ("fc1", model.head[1], "head.flatten", "head.fc1_out"),
        ("fc2", model.head[4], "head.gelu1", "head.fc2_out"),
        ("fc3", model.head[7], "head.gelu2", "head.fc3_out"),
    ]
    for name, linear, input_key, output_key in linear_specs:
        q_weight, weight_scales = quantize_weight_per_output_channel(linear.weight)
        bias = linear.bias.detach().float().cpu() if linear.bias is not None else torch.zeros(linear.weight.shape[0])
        q_bias, bias_scales = quantize_bias_int32(bias, activation_scales[input_key], weight_scales)
        arrays[f"{name}.weight_int8"] = q_weight
        arrays[f"{name}.weight_scale"] = weight_scales
        arrays[f"{name}.bias_int32"] = q_bias
        arrays[f"{name}.bias_scale"] = bias_scales
        layers[name] = _layer_metadata(
            layer_type="linear",
            weight_shape=list(q_weight.shape),
            input_scale=activation_scales[input_key],
            output_scale=activation_scales[output_key],
            weight_scales=weight_scales,
            extra={"input_scale_key": input_key, "output_scale_key": output_key},
        )

    luts = {
        "encoder.gelu1": {
            "input_scale_key": "encoder.conv1_out",
            "output_scale_key": "encoder.gelu1",
            "array": "lut.encoder.gelu1.int8",
        },
        "encoder.gelu2": {
            "input_scale_key": "encoder.conv2_out",
            "output_scale_key": "encoder.gelu2",
            "array": "lut.encoder.gelu2.int8",
        },
        "head.gelu1": {
            "input_scale_key": "head.fc1_out",
            "output_scale_key": "head.gelu1",
            "array": "lut.head.gelu1.int8",
        },
        "head.gelu2": {
            "input_scale_key": "head.fc2_out",
            "output_scale_key": "head.gelu2",
            "array": "lut.head.gelu2.int8",
        },
    }
    for lut in luts.values():
        arrays[lut["array"]] = gelu_lut(
            activation_scales[lut["input_scale_key"]],
            activation_scales[lut["output_scale_key"]],
        )
        lut["index"] = "signed_int8_code_plus_128"

    pool_scale = activation_scales["encoder.gelu2"] / activation_scales["encoder.pool"]
    pool_multiplier, pool_shift = quantize_multiplier(pool_scale)
    metadata = {
        "format_version": 1,
        "model_name": "multi_esp_fast_pose_cnn",
        "int8_range": [INT8_QMIN, INT8_QMAX],
        "node_count": int(model.node_count),
        "receiver_feature_dim": 32,
        "layers": layers,
        "luts": luts,
        "pool": {
            "type": "adaptive_avg_pool2d",
            "output_size": [8, 4],
            "input_scale_key": "encoder.gelu2",
            "output_scale_key": "encoder.pool",
            "requant_multiplier": pool_multiplier,
            "requant_shift": pool_shift,
            "real_scale_without_region_divisor": pool_scale,
        },
        "activation_scales": activation_scales,
        "calibration": calibration_report,
        "arrays": sorted(arrays),
    }
    np.savez_compressed(output_dir / "weights_int8.npz", **arrays)
    (output_dir / "model_int8.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
