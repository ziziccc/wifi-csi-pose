# ML Fast CNN

This directory contains the Fast CNN pose model and an FPGA-oriented INT8 export/reference path.

## Main Files

- `configs/multi_esp_fast_pose_cnn.json`: float32 Fast CNN training config
- `configs/fast_cnn_int8.json`: INT8 export/reference config
- `src/original_models.py`: `MultiESPFastPoseCNN`
- `src/int8_export.py`: INT8 weight, INT32 bias, scale, requant, and LUT export
- `src/int8_reference.py`: true integer INT8 reference inference
- `src/training.py`: Fast CNN training and evaluation
- `src/evaluation.py`: evaluation helpers
- `src/data.py`: cache dataset
- `src/prepare.py`: CSV to cache conversion
- `scripts/run.py`: prepare/train/evaluate workflow
- `scripts/view_gui.py`: Fast CNN pose viewer
- `scripts/infer_gui.py`: selected CSV cache + viewer wrapper
- `scripts/export_int8_fast_cnn.py`: checkpoint to FPGA INT8 artifacts
- `scripts/run_int8_fast_cnn.py`: run true-integer INT8 reference and compare with float32

## Float32 Fast CNN

```powershell
python scripts/run.py --skip-prepare
```

View an existing checkpoint:

```powershell
python scripts/view_gui.py --config .\configs\multi_esp_fast_pose_cnn.json
```

If `outputs/int8_fast_cnn` exists, the viewer overlays target, float32 Fast CNN, and true-integer INT8 reference outputs. To disable the INT8 overlay:

```powershell
python scripts/view_gui.py --config .\configs\multi_esp_fast_pose_cnn.json --no-int8
```

View selected CSV inputs:

```powershell
python scripts/infer_gui.py .\infer_csv
```

`infer_gui.py` also opens the same overlay viewer. It passes `outputs/int8_fast_cnn` as the default INT8 artifact directory. To disable the INT8 overlay:

```powershell
python scripts/infer_gui.py .\infer_csv --no-int8
```

## FPGA INT8 Export

Export FPGA-oriented INT8 artifacts:

```powershell
python scripts/export_int8_fast_cnn.py --config .\configs\fast_cnn_int8.json
```

Generated files:

```text
outputs/int8_fast_cnn/weights_int8.npz
outputs/int8_fast_cnn/model_int8.json
```

Run the true-integer PC reference:

```powershell
python scripts/run_int8_fast_cnn.py --config .\configs\fast_cnn_int8.json --windows 8 --dump .\outputs\int8_fast_cnn\layer_dump_sample.npz
```

The reference path uses `int8 activation * int8 weight -> int32 accumulate -> integer requantize`.
It does not use fake-quantized float Conv/Linear operations.
