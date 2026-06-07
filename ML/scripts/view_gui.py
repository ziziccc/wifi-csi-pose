from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
import torch


JOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]
PLAYBACK_SPEED_OPTIONS = ("0.25x", "0.5x", "1.0x", "2.0x", "4.0x", "8.0x", "16.0x")
BASE_PLAYBACK_DELAY_MS = 180


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_path() -> None:
    src_dir = _project_dir() / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _resolve(base_dir: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def build_parser() -> argparse.ArgumentParser:
    project_dir = _project_dir()
    parser = argparse.ArgumentParser(description="View Fast CNN pose predictions.")
    parser.add_argument("--config", type=Path, default=project_dir / "configs" / "multi_esp_fast_pose_cnn.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--windows", type=int, default=0, help="0 means every valid window in the split.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--int8-dir", type=Path, default=project_dir / "outputs" / "int8_fast_cnn")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 reference overlay.")
    parser.add_argument("--log-interval", type=int, default=1, help="Print viewer prediction progress every N batches.")
    return parser


def _load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_predictions(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    _ensure_src_path()
    from data import CachedWindowDataset
    from original_models import build_model_from_checkpoint
    if not args.no_int8:
        from int8_reference import fast_cnn_int8_forward, load_int8_artifacts

    config_path = args.config.resolve()
    config = _load_config(config_path)
    started_at = time.perf_counter()
    print("[view] loading config/checkpoint/cache settings", flush=True)
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.resolve()
    else:
        cache_dir = _resolve(config_path.parent, config["cache_dir"])
        fallback_cache_dir = _project_dir() / "outputs" / "selected_csv_cache"
        if not cache_dir.exists() and fallback_cache_dir.exists():
            cache_dir = fallback_cache_dir
    output_dir = _resolve(config_path.parent, config["output_dir"])
    checkpoint_path = args.checkpoint.resolve() if args.checkpoint is not None else output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fast CNN checkpoint not found: {checkpoint_path}")
    print(f"[view] checkpoint={checkpoint_path}", flush=True)
    print(f"[view] cache_dir={cache_dir} split={args.split}", flush=True)

    dataset = CachedWindowDataset(
        npz_path=cache_dir / f"{args.split}.npz",
        window_size=int(config.get("window_size", 10)),
        window_stride=int(config.get("window_stride", 1)),
        feature_mode=str(config.get("feature_mode", "all")),
        require_full_window_mask=bool(config.get("require_full_window_mask", False)),
        fill_mode=str(config.get("fill_mode", "forward_fill")),
        max_gap=int(config.get("max_gap", 3)),
        return_prev_target=False,
        return_file_id=False,
        motion_lag=1,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No valid windows found in {cache_dir / f'{args.split}.npz'}.")

    limit = int(args.windows)
    indices = list(range(len(dataset))) if limit <= 0 else list(range(min(limit, len(dataset))))
    batch_size = int(args.batch_size or config.get("batch_size", 64))
    log_interval = max(int(args.log_interval), 1)
    print(f"[view] valid_windows={len(dataset)} selected_windows={len(indices)} batch_size={batch_size}", flush=True)
    print("[view] loading float32 model", flush=True)
    model, checkpoint = build_model_from_checkpoint(str(checkpoint_path))
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    int8_arrays = None
    int8_metadata = None
    int8_dir = args.int8_dir.resolve()
    if not args.no_int8:
        if not (int8_dir / "weights_int8.npz").exists() or not (int8_dir / "model_int8.json").exists():
            raise FileNotFoundError(
                f"INT8 artifacts not found in {int8_dir}. Run scripts/export_int8_fast_cnn.py first or pass --no-int8."
            )
        print(f"[view] loading INT8 artifacts={int8_dir}", flush=True)
        int8_arrays, int8_metadata = load_int8_artifacts(int8_dir)
    else:
        print("[view] INT8 overlay disabled", flush=True)

    predictions: list[np.ndarray] = []
    int8_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    batch_total = (len(indices) + batch_size - 1) // batch_size
    print("[view] running float32/INT8 predictions before opening GUI", flush=True)
    with torch.no_grad():
        for batch_number, start in enumerate(range(0, len(indices), batch_size), start=1):
            batch_indices = indices[start : start + batch_size]
            should_log_batch = batch_number == 1 or batch_number == batch_total or batch_number % log_interval == 0
            if should_log_batch:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[view] batch {batch_number}/{batch_total} start "
                    f"indices={batch_indices[0]}..{batch_indices[-1]} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            batch_inputs = []
            batch_targets = []
            for index in batch_indices:
                inputs, target = dataset[index]
                batch_inputs.append(inputs)
                batch_targets.append(target)
            input_tensor_cpu = torch.stack(batch_inputs).to(dtype=torch.float32)
            input_tensor = input_tensor_cpu.to(device=device)
            prediction = model(input_tensor).cpu().numpy().astype(np.float32)
            if should_log_batch:
                elapsed = time.perf_counter() - started_at
                print(f"[view] batch {batch_number}/{batch_total} float32 done elapsed={elapsed:.1f}s", flush=True)
            if int8_arrays is not None and int8_metadata is not None:
                int8_prediction, _ = fast_cnn_int8_forward(
                    input_tensor_cpu.numpy().astype(np.float32),
                    int8_arrays,
                    int8_metadata,
                    dump_intermediates=False,
                )
                int8_predictions.append(int8_prediction.astype(np.float32))
                if should_log_batch:
                    elapsed = time.perf_counter() - started_at
                    print(f"[view] batch {batch_number}/{batch_total} INT8 done elapsed={elapsed:.1f}s", flush=True)
            target_array = torch.stack(batch_targets).numpy().astype(np.float32)
            predictions.append(prediction)
            targets.append(target_array)
            if should_log_batch:
                elapsed = time.perf_counter() - started_at
                print(
                    f"[view] batch {batch_number}/{batch_total} "
                    f"windows={min(start + batch_size, len(indices))}/{len(indices)} "
                    f"stored elapsed={elapsed:.1f}s",
                    flush=True,
                )

    prediction_array = np.concatenate(predictions, axis=0)
    int8_prediction_array = np.concatenate(int8_predictions, axis=0) if int8_predictions else None
    target_array = np.concatenate(targets, axis=0)
    metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "cache_dir": str(cache_dir),
        "int8_dir": str(int8_dir) if int8_prediction_array is not None else None,
        "split": args.split,
        "windows": int(prediction_array.shape[0]),
        "model_name": str(checkpoint.get("model_name", "multi_esp_fast_pose_cnn")),
    }
    print(f"[view] prediction arrays ready in {time.perf_counter() - started_at:.1f}s; opening GUI", flush=True)
    return prediction_array, int8_prediction_array, target_array, metadata


class FastPoseViewer(tk.Tk):
    def __init__(
        self,
        predictions: np.ndarray,
        int8_predictions: np.ndarray | None,
        targets: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__()
        self.predictions = predictions.reshape(predictions.shape[0], 12, 2)
        self.int8_predictions = (
            int8_predictions.reshape(int8_predictions.shape[0], 12, 2)
            if int8_predictions is not None
            else None
        )
        self.targets = targets.reshape(targets.shape[0], 12, 2)
        self.metadata = metadata
        self.current_index = 0
        self.is_playing = False
        self.playback_after_id: str | None = None
        self.canvas_width = 820
        self.canvas_height = 620

        self.title("ML2 Fast CNN Pose Viewer")
        self.geometry("1180x820")
        self.minsize(980, 700)

        self.index_var = tk.IntVar(value=0)
        self.index_label_var = tk.StringVar(value="")
        self.play_button_var = tk.StringVar(value="Play")
        self.playback_speed_var = tk.StringVar(value="1.0x")
        self.show_target_var = tk.BooleanVar(value=True)
        self.show_prediction_var = tk.BooleanVar(value=True)
        self.show_int8_var = tk.BooleanVar(value=self.int8_predictions is not None)
        self.summary_var = tk.StringVar(value="")

        self._build_ui()
        self._render_current_sample()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=(12, 12, 12, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        controls = ttk.Frame(top)
        controls.grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="-100", command=lambda: self._step_index(-100)).grid(row=0, column=0, padx=2)
        ttk.Button(controls, text="-10", command=lambda: self._step_index(-10)).grid(row=0, column=1, padx=2)
        ttk.Button(controls, text="Prev", command=lambda: self._step_index(-1)).grid(row=0, column=2, padx=2)
        ttk.Button(controls, text="Next", command=lambda: self._step_index(1)).grid(row=0, column=3, padx=2)
        ttk.Button(controls, text="+10", command=lambda: self._step_index(10)).grid(row=0, column=4, padx=2)
        ttk.Button(controls, text="+100", command=lambda: self._step_index(100)).grid(row=0, column=5, padx=2)
        ttk.Button(controls, textvariable=self.play_button_var, command=self._toggle_playback).grid(row=0, column=6, padx=(10, 2))
        ttk.Label(controls, text="Speed").grid(row=0, column=7, padx=(10, 4), sticky="e")
        speed_combo = ttk.Combobox(
            controls,
            textvariable=self.playback_speed_var,
            values=PLAYBACK_SPEED_OPTIONS,
            state="readonly",
            width=7,
        )
        speed_combo.grid(row=0, column=8, padx=2)

        toggles = ttk.Frame(top)
        toggles.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(toggles, text="Target", variable=self.show_target_var, command=self._render_current_sample).grid(row=0, column=0, padx=4)
        ttk.Checkbutton(toggles, text="Float32", variable=self.show_prediction_var, command=self._render_current_sample).grid(row=0, column=1, padx=4)
        int8_check = ttk.Checkbutton(toggles, text="INT8", variable=self.show_int8_var, command=self._render_current_sample)
        int8_check.grid(row=0, column=2, padx=4)
        if self.int8_predictions is None:
            int8_check.state(["disabled"])
        ttk.Label(toggles, textvariable=self.index_label_var).grid(row=0, column=3, padx=(16, 4))

        scale = ttk.Scale(
            top,
            from_=0,
            to=max(len(self.predictions) - 1, 0),
            orient="horizontal",
            variable=self.index_var,
            command=self._on_scale_changed,
        )
        scale.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(body, width=self.canvas_width, height=self.canvas_height, background="#000000", highlightthickness=1, highlightbackground="#334155")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(body, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")
        ttk.Label(side, text="Summary").grid(row=0, column=0, sticky="w")
        ttk.Label(side, textvariable=self.summary_var, justify="left", width=42).grid(row=1, column=0, sticky="nw", pady=(8, 0))

    def _on_scale_changed(self, _: str) -> None:
        self.current_index = int(round(self.index_var.get()))
        self._render_current_sample()

    def _step_index(self, delta: int) -> None:
        self.current_index = min(max(self.current_index + delta, 0), len(self.predictions) - 1)
        self.index_var.set(self.current_index)
        self._render_current_sample()

    def _toggle_playback(self) -> None:
        self.is_playing = not self.is_playing
        self.play_button_var.set("Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._schedule_next_frame()
        elif self.playback_after_id is not None:
            self.after_cancel(self.playback_after_id)
            self.playback_after_id = None

    def _schedule_next_frame(self) -> None:
        if not self.is_playing:
            return
        speed = float(self.playback_speed_var.get().rstrip("x"))
        delay_ms = max(int(BASE_PLAYBACK_DELAY_MS / max(speed, 0.01)), 1)
        self.playback_after_id = self.after(delay_ms, self._advance_playback)

    def _advance_playback(self) -> None:
        self.current_index = (self.current_index + 1) % len(self.predictions)
        self.index_var.set(self.current_index)
        self._render_current_sample()
        self._schedule_next_frame()

    def _project_points(self, points: np.ndarray, reference_points: np.ndarray) -> list[tuple[float, float]]:
        finite = np.isfinite(reference_points).all(axis=1)
        margin = 24.0
        draw_width = max(self.canvas_width - (2.0 * margin), 1.0)
        draw_height = max(self.canvas_height - (2.0 * margin), 1.0)
        if finite.any():
            valid = reference_points[finite]
            if float(valid.min()) >= -0.25 and float(valid.max()) <= 1.25:
                return [
                    (
                        float(margin + (x * draw_width)),
                        float(margin + (y * draw_height)),
                    )
                    for x, y in points
                ]

        if not finite.any():
            return [(self.canvas_width / 2, self.canvas_height / 2)] * len(points)
        valid = reference_points[finite]
        min_xy = valid.min(axis=0)
        max_xy = valid.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        span = np.maximum(max_xy - min_xy, 1.0e-6)
        scale = 0.78 * min(self.canvas_width / span[0], self.canvas_height / span[1])
        projected = []
        for x, y in points:
            px = self.canvas_width / 2 + (x - center[0]) * scale
            py = self.canvas_height / 2 + (y - center[1]) * scale
            projected.append((float(px), float(py)))
        return projected

    def _draw_pose(
        self,
        points: np.ndarray,
        *,
        reference_points: np.ndarray,
        color: str,
        label: str,
        label_y: int,
        offset: float = 0.0,
    ) -> None:
        projected = [(x + offset, y) for x, y in self._project_points(points, reference_points)]
        name_to_index = {name: index for index, name in enumerate(JOINT_NAMES)}
        for start_name, end_name in SKELETON_EDGES:
            x1, y1 = projected[name_to_index[start_name]]
            x2, y2 = projected[name_to_index[end_name]]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=3)
        for x, y in projected:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="")
        self.canvas.create_text(18, label_y, text=label, fill=color, anchor="w", font=("Segoe UI", 11, "bold"))

    def _render_current_sample(self) -> None:
        self.canvas.delete("all")
        target = self.targets[self.current_index]
        prediction = self.predictions[self.current_index]
        int8_prediction = self.int8_predictions[self.current_index] if self.int8_predictions is not None else None
        reference_parts = [target, prediction]
        if int8_prediction is not None:
            reference_parts.append(int8_prediction)
        reference_points = np.concatenate(reference_parts, axis=0)
        if self.show_target_var.get():
            self._draw_pose(target, reference_points=reference_points, color="#38bdf8", label="Target", label_y=22, offset=0.0)
        if self.show_prediction_var.get():
            self._draw_pose(prediction, reference_points=reference_points, color="#f97316", label="Float32", label_y=44, offset=0.0)
        if int8_prediction is not None and self.show_int8_var.get():
            self._draw_pose(int8_prediction, reference_points=reference_points, color="#22c55e", label="INT8", label_y=66, offset=0.0)

        abs_error = np.abs(prediction.reshape(-1) - target.reshape(-1))
        summary_lines = [
            f"model: {self.metadata['model_name']}",
            f"split: {self.metadata['split']}",
            f"windows: {self.metadata['windows']}",
            f"float_mae: {float(abs_error.mean()):.6f}",
            f"float_max_abs_error: {float(abs_error.max()):.6f}",
        ]
        if int8_prediction is not None:
            int8_abs_error = np.abs(int8_prediction.reshape(-1) - target.reshape(-1))
            int8_float_diff = np.abs(int8_prediction.reshape(-1) - prediction.reshape(-1))
            summary_lines.extend(
                [
                    f"int8_mae: {float(int8_abs_error.mean()):.6f}",
                    f"int8_vs_float_mae: {float(int8_float_diff.mean()):.6f}",
                    f"int8_vs_float_max: {float(int8_float_diff.max()):.6f}",
                    f"int8: {self.metadata['int8_dir']}",
                ]
            )
        summary_lines.extend(
            [
                f"checkpoint: {self.metadata['checkpoint']}",
                f"cache: {self.metadata['cache_dir']}",
            ]
        )
        self.index_label_var.set(f"{self.current_index + 1} / {len(self.predictions)}")
        self.summary_var.set("\n".join(summary_lines))


def main() -> int:
    args = build_parser().parse_args()
    try:
        predictions, int8_predictions, targets, metadata = _load_predictions(args)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Fast CNN viewer error", str(exc))
        return 1
    app = FastPoseViewer(predictions, int8_predictions, targets, metadata)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
