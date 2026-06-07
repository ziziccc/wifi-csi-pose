# Fast CNN Model

The model class is `MultiESPFastPoseCNN`. It converts a CSI window directly into a 24-value pose vector.

## Input

The dataset passes each window as:

```text
[batch, node_count * 3, subcarriers, window_size]
```

Default shape:

```text
node_count = 3
feature groups = base, delta, mask
input channels = 9
subcarriers = 128
window_size = 10
```

Each node is sliced into `base`, `delta`, and `mask` channels. The same receiver encoder is shared by all nodes.

## Encoder

For one node, input shape is `[batch, 3, 128, 10]`.

```text
Conv2d(3 -> 16, kernel=(5,3), padding=(2,1), bias=False)
BatchNorm2d(16)
GELU
Conv2d(16 -> 32, kernel=(3,3), padding=1, bias=False)
BatchNorm2d(32)
GELU
AdaptiveAvgPool2d((8,4))
```

One node becomes `[batch, 32, 8, 4]`. With three nodes, concat gives `[batch, 96, 8, 4]`; flatten gives 3072 values.

## Head

```text
Flatten
Linear(3072 -> 128)
GELU
Dropout(0.15)
Linear(128 -> 128)
GELU
Dropout(0.15)
Linear(128 -> 24)
```

The final 24 values are 12 `(x, y)` keypoints.

## FPGA INT8 Path

INT8 export is implemented by `src/int8_export.py` and exposed through `scripts/export_int8_fast_cnn.py`.

Conv layers are exported after folding BatchNorm into Conv:

```text
bn_scale[o] = gamma[o] / sqrt(running_var[o] + eps)
W_fused[o] = W_conv[o] * bn_scale[o]
b_fused[o] = beta[o] + (0 - running_mean[o]) * bn_scale[o]
```

Weights use symmetric per-output-channel INT8 quantization:

```text
weight_scale[o] = max(abs(W_fused[o])) / 127
weight_int8[o] = clamp(round(W_fused[o] / weight_scale[o]), -127, 127)
```

Bias is INT32:

```text
bias_scale[o] = input_scale * weight_scale[o]
bias_int32[o] = round(b_fused[o] / bias_scale[o])
```

Conv/Linear integer MAC:

```text
acc_int32[o] = sum(input_int8[i] * weight_int8[o, i]) + bias_int32[o]
```

Requantization to the next INT8 activation:

```text
real_scale[o] = input_scale * weight_scale[o] / output_scale
output_int8[o] = clamp(round(acc_int32[o] * real_scale[o]), -127, 127)
```

`model_int8.json` stores integer multiplier/shift approximations:

```text
real_scale[o] ~= requant_multiplier[o] / 2^requant_shift[o]
```

FPGA/reference computation:

```text
tmp = acc_int32[o] * requant_multiplier[o]
rounded = round_shift(tmp, requant_shift[o])
output_int8[o] = clamp(rounded, -127, 127)
```

GELU is exported as a 256-entry INT8 LUT:

```text
gelu_out_int8 = lut[input_int8 + 128]
```

AdaptiveAvgPool is computed with INT32 sums:

```text
sum_int32 = sum(region_int8)
pool_int8 = clamp(round(sum_int32 * gelu2_scale / (region_size * pool_scale)), -127, 127)
```

`src/int8_reference.py` is the PC-side reference for this FPGA integer path. It does not use fake quant or float Conv/Linear operations.
