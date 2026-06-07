from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import numpy as np


SERIAL_MAGIC = 0x35534943
FRAME_CYCLE = 2
SERIAL_HEADER = struct.Struct("<IBBHII")
CYCLE_HEADER = struct.Struct("<IIQQIBBBB")
CYCLE_SLOT = struct.Struct("<BBbBH")
MAX_FRAME_PAYLOAD = 12_288
POSE_DIM = 24

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


@dataclass
class RunResult:
    csv_path: Path
    model_name: str
    trigger_seq: np.ndarray
    target_pose: np.ndarray
    pc_float32_pose: np.ndarray
    pc_int8_pose: np.ndarray
    pc_int8_codes: np.ndarray
    fpga_pose: np.ndarray
    fpga_timing: list[dict[str, float]]
    pc_float32_elapsed_sec: float
    pc_int8_elapsed_sec: float
    pc_float32_forward_ms: np.ndarray
    pc_int8_forward_ms: np.ndarray
    fpga_elapsed_sec: float
    output_scale: float
    ps_float32_pose: np.ndarray | None = None
    ps_int8_pose: np.ndarray | None = None
    ps_float32_forward_ms: np.ndarray | None = None
    ps_int8_forward_ms: np.ndarray | None = None
    output_npz: Path | None = None
    summary_json: Path | None = None

    @property
    def sample_count(self) -> int:
        return int(min(len(self.pc_float32_pose), len(self.pc_int8_pose), len(self.fpga_pose), len(self.target_pose)))


class RunCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[Any]] = set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            processes = list(self._processes)
        for proc in processes:
            terminate_process(proc)

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.is_cancelled():
            raise RunCancelled("Run stopped by user.")

    def register(self, proc: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.add(proc)
        if self.is_cancelled():
            terminate_process(proc)

    def unregister(self, proc: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.discard(proc)


def terminate_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=1.0)
    except Exception:  # noqa: BLE001
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def cancel_check(cancel: CancellationToken | None) -> None:
    if cancel is not None:
        cancel.check()


def run_subprocess(
    command: list[str],
    *,
    cancel: CancellationToken | None = None,
    input: bytes | str | None = None,  # noqa: A002
    capture_output: bool = False,
    check: bool = False,
    text: bool | None = None,
    cwd: str | Path | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> subprocess.CompletedProcess[Any]:
    if cancel is None:
        return subprocess.run(
            command,
            input=input,
            capture_output=capture_output,
            check=check,
            text=text,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
        )

    cancel.check()
    if capture_output:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if input is not None else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
        cwd=cwd,
    )
    cancel.register(proc)
    try:
        out, err = proc.communicate(input=input)
    finally:
        cancel.unregister(proc)
    cancel.check()
    result = subprocess.CompletedProcess(command, proc.returncode, out, err)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, command, output=out, stderr=err)
    return result


def start_subprocess(
    command: list[str],
    *,
    cancel: CancellationToken | None = None,
    **kwargs: Any,
) -> subprocess.Popen[Any]:
    cancel_check(cancel)
    proc = subprocess.Popen(command, **kwargs)
    if cancel is not None:
        cancel.register(proc)
    return proc


def pkill_no_self_pattern(text: str) -> str:
    if not text:
        return r"$^"
    return f"[{re.escape(text[0])}]{re.escape(text[1:])}"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_ml_root() -> Path:
    return project_root() / "ML"


def tool_root() -> Path:
    return Path(__file__).resolve().parent


def ensure_ml_paths(ml_root: Path) -> None:
    for path in (ml_root / "src", ml_root / "scripts", ml_root / "scripts" / "tools"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base_dir: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def checksum32(data: bytes, seed: int = 0) -> int:
    value = seed
    for byte in data:
        value = ((value << 5) - value + byte) & 0xFFFFFFFF
    return value


def encode_serial_frame(frame_type: int, payload: bytes, frame_seq: int) -> bytes:
    if len(payload) > MAX_FRAME_PAYLOAD:
        raise ValueError(f"serial payload too large: {len(payload)} > {MAX_FRAME_PAYLOAD}")
    header = SERIAL_HEADER.pack(SERIAL_MAGIC, 1, frame_type, len(payload), frame_seq, 0)
    cksum = checksum32(header[: SERIAL_HEADER.size - 4] + payload)
    return SERIAL_HEADER.pack(SERIAL_MAGIC, 1, frame_type, len(payload), frame_seq, cksum) + payload


def iq_pairs_to_csi_bytes(raw_pairs: str) -> bytes:
    pairs = json.loads(raw_pairs or "[]")
    flat: list[int] = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        i_val = max(-128, min(127, int(pair[0])))
        q_val = max(-128, min(127, int(pair[1])))
        flat.extend([i_val, q_val])
    return struct.pack(f"<{len(flat)}b", *flat) if flat else b""


def csv_to_serial_frames(csv_path: Path, *, node_count: int, slot_timeout_us: int = 1000) -> tuple[bytes, list[int]]:
    frames = bytearray()
    trigger_order: list[int] = []
    frame_seq = 0
    uart_seq = 0

    def flush_group(trigger_seq: int, rows: list[dict[str, str]]) -> None:
        nonlocal frame_seq, uart_seq
        by_rx: dict[int, dict[str, str]] = {}
        for row in rows:
            try:
                rx_index = int(row.get("rx_index", ""))
            except ValueError:
                continue
            if 0 <= rx_index < node_count and row.get("iq_pairs"):
                by_rx[rx_index] = row

        if not by_rx:
            return

        uart_seq += 1
        trigger_order.append(trigger_seq)
        payload = bytearray(
            CYCLE_HEADER.pack(
                uart_seq,
                trigger_seq,
                0,
                0,
                slot_timeout_us,
                node_count,
                len(by_rx),
                0 if len(by_rx) == node_count else 1,
                0,
            )
        )
        for rx_index in range(node_count):
            row = by_rx.get(rx_index)
            if row is None:
                payload.extend(CYCLE_SLOT.pack(rx_index, 0, 0, 0, 0))
                continue
            csi_bytes = iq_pairs_to_csi_bytes(row.get("iq_pairs", ""))
            rssi = int(float(row.get("rssi", "0") or 0))
            rssi = max(-128, min(127, rssi))
            payload.extend(CYCLE_SLOT.pack(rx_index, 1, rssi, 0, len(csi_bytes)))
            payload.extend(csi_bytes)

        frame_seq += 1
        frames.extend(encode_serial_frame(FRAME_CYCLE, bytes(payload), frame_seq))

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        current_trigger: int | None = None
        group_rows: list[dict[str, str]] = []
        for row in reader:
            raw_trigger = row.get("trigger_seq")
            if raw_trigger is None or raw_trigger == "":
                continue
            trigger_seq = int(raw_trigger)
            if current_trigger is None:
                current_trigger = trigger_seq
            if trigger_seq != current_trigger:
                flush_group(current_trigger, group_rows)
                group_rows = []
                current_trigger = trigger_seq
            group_rows.append(row)
        if current_trigger is not None:
            flush_group(current_trigger, group_rows)

    return bytes(frames), trigger_order


def resolve_input(input_path: Path, pattern: str) -> tuple[Path, str, list[Path]]:
    input_path = input_path.resolve()
    if input_path.is_file():
        files = [input_path]
        return input_path.parent, input_path.name, files
    files = sorted(path for path in input_path.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_path} with pattern {pattern!r}")
    return input_path, pattern, files


def prepare_csv_cache(
    *,
    ml_root: Path,
    input_path: Path,
    pattern: str,
    cache_dir: Path,
    node_count: int,
    pair_count: int,
    subcarrier_remap: str,
) -> None:
    ensure_ml_paths(ml_root)
    from infer_gui import _prepare_test_cache

    input_dir, input_pattern, _ = resolve_input(input_path, pattern)
    _prepare_test_cache(
        input_dir=input_dir,
        pattern=input_pattern,
        cache_dir=cache_dir,
        node_count=node_count,
        pair_count=pair_count,
        subcarrier_remap=subcarrier_remap,
    )


def prepared_window_count(*, ml_root: Path, config_path: Path, cache_dir: Path) -> int:
    ensure_ml_paths(ml_root)
    from data import CachedWindowDataset

    config = load_json(config_path)
    dataset = CachedWindowDataset(
        cache_dir / "test.npz",
        window_size=int(config.get("window_size", 10)),
        window_stride=int(config.get("window_stride", 10)),
        feature_mode="all",
        require_full_window_mask=False,
        fill_mode="forward_fill",
        max_gap=3,
        return_prev_target=False,
        return_file_id=False,
        motion_lag=1,
    )
    return len(dataset)


def prepare_and_count_windows(args: argparse.Namespace) -> int:
    prepare_csv_cache(
        ml_root=args.ml_root.resolve(),
        input_path=args.csv.resolve(),
        pattern=args.pattern,
        cache_dir=args.cache_dir.resolve(),
        node_count=args.node_count,
        pair_count=args.pair_count,
        subcarrier_remap=args.subcarrier_remap,
    )
    return prepared_window_count(
        ml_root=args.ml_root.resolve(),
        config_path=args.config.resolve(),
        cache_dir=args.cache_dir.resolve(),
    )


def cache_is_ready(cache_dir: Path, input_path: Path, pattern: str) -> bool:
    metadata_path = cache_dir / "metadata.json"
    if not (cache_dir / "test.npz").exists() or not metadata_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
        input_dir, input_pattern, files = resolve_input(input_path, pattern)
        cached_files = [str(Path(path).resolve()) for path in metadata.get("splits", {}).get("test", {}).get("files", [])]
        current_files = [str(path.resolve()) for path in files]
        return (
            str(Path(metadata.get("input_dir", "")).resolve()) == str(input_dir.resolve())
            and str(metadata.get("pattern", "")) == input_pattern
            and cached_files == current_files
        )
    except Exception:
        return False


def selected_file_window_counts(*, ml_root: Path, config_path: Path, cache_dir: Path, windows: int) -> dict[int, int]:
    ensure_ml_paths(ml_root)
    from data import CachedWindowDataset
    from evaluation import checkpoint_data_settings

    config = load_json(config_path)
    checkpoint_path = resolve_path(config_path.parent, config["checkpoint"])
    if checkpoint_path is None:
        raise ValueError("config checkpoint is required")
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    settings = checkpoint_data_settings(checkpoint)
    settings["window_size"] = int(config.get("window_size", settings["window_size"]))
    settings["window_stride"] = int(config.get("window_stride", settings["window_stride"]))
    dataset = CachedWindowDataset(cache_dir / "test.npz", **settings)
    limit = len(dataset) if windows <= 0 else min(windows, len(dataset))
    raw = np.load(cache_dir / "test.npz")
    counts: dict[int, int] = {}
    for dataset_index in range(limit):
        raw_index = dataset.indices[dataset_index]
        file_id = int(raw["file_ids"][raw_index])
        counts[file_id] = counts.get(file_id, 0) + 1
    return counts


def output_scale_key(model_name: str) -> str:
    if model_name == "multi_esp_fast_pose_cnn":
        return "head.fc3_out"
    raise ValueError(f"Unsupported model_name={model_name!r}")


def ensure_int8_export(ml_root: Path, config_path: Path, cancel: CancellationToken | None = None) -> Path:
    config = load_json(config_path)
    output_dir = resolve_path(config_path.parent, config["output_dir"])
    if output_dir is None:
        raise ValueError("config output_dir is required")
    weights_path = output_dir / "weights_int8.npz"
    model_json_path = output_dir / "model_int8.json"
    if weights_path.exists() and model_json_path.exists():
        return output_dir
    command = [sys.executable, str(ml_root / "scripts" / "export_int8_fast_cnn.py"), "--config", str(config_path)]
    run_subprocess(command, cwd=ml_root, check=True, cancel=cancel)
    if not weights_path.exists() or not model_json_path.exists():
        raise FileNotFoundError(f"INT8 artifacts were not created in {output_dir}")
    return output_dir


def run_pc_inference(
    *,
    ml_root: Path,
    config_path: Path,
    cache_dir: Path,
    windows: int,
    batch_size: int | None,
    device_text: str,
    cancel: CancellationToken | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray, float]:
    ensure_ml_paths(ml_root)
    import torch
    from torch.utils.data import DataLoader, Subset

    from data import CachedWindowDataset
    from evaluation import checkpoint_data_settings
    from int8_reference import fast_cnn_int8_forward, load_int8_artifacts
    from original_models import build_model_from_checkpoint

    config = load_json(config_path)
    checkpoint_path = resolve_path(config_path.parent, config["checkpoint"])
    if checkpoint_path is None:
        raise ValueError("config checkpoint is required")
    cancel_check(cancel)
    int8_dir = ensure_int8_export(ml_root, config_path, cancel=cancel)
    int8_arrays, int8_metadata = load_int8_artifacts(int8_dir)
    scale = float(int8_metadata["activation_scales"][output_scale_key(str(int8_metadata["model_name"]))])

    float_model, checkpoint = build_model_from_checkpoint(checkpoint_path)
    settings = checkpoint_data_settings(checkpoint)
    settings["window_size"] = int(config.get("window_size", settings["window_size"]))
    settings["window_stride"] = int(config.get("window_stride", settings["window_stride"]))

    dataset = CachedWindowDataset(cache_dir / "test.npz", **settings)
    indices = list(range(len(dataset)))
    if windows > 0:
        indices = indices[: min(windows, len(indices))]
    subset = Subset(dataset, indices)
    realtime_batch_size = 1
    loader = DataLoader(
        subset,
        batch_size=realtime_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    raw_npz = np.load(cache_dir / "test.npz")
    raw_end_indices = np.asarray([dataset.indices[index] for index in indices], dtype=np.int64)
    trigger_seq = raw_npz["trigger_seq"][raw_end_indices].astype(np.int64) if len(raw_end_indices) else np.zeros(0, dtype=np.int64)

    device = torch.device(device_text)
    float_model.eval().to(device)

    def move_targets(targets: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(targets, dict):
            return {key: value.to(device) for key, value in targets.items()}
        return targets.to(device)

    def pose(output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            return output["pose"]
        return output

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def infer_model(model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        outputs: list[np.ndarray] = []
        targets_out: list[np.ndarray] = []
        forward_ms: list[float] = []
        prev_pose: torch.Tensor | None = None
        prev_file_id: int | None = None
        started = time.perf_counter()
        with torch.no_grad():
            for inputs, targets in loader:
                cancel_check(cancel)
                inputs = inputs.to(device)
                targets = move_targets(targets)
                if getattr(model, "use_prev_pose", False):
                    batch_outputs: list[torch.Tensor | dict[str, torch.Tensor]] = []
                    file_ids = targets.get("file_id") if isinstance(targets, dict) else None
                    for sample_index in range(inputs.shape[0]):
                        current_file_id = int(file_ids[sample_index].item()) if file_ids is not None else prev_file_id
                        if prev_file_id is None or (current_file_id is not None and current_file_id != prev_file_id):
                            sample_prev_pose = None
                        else:
                            sample_prev_pose = prev_pose

                        synchronize()
                        forward_started = time.perf_counter()
                        sample_output = model(inputs[sample_index : sample_index + 1], prev_pose=sample_prev_pose)
                        synchronize()
                        forward_ms.append((time.perf_counter() - forward_started) * 1000.0)
                        batch_outputs.append(sample_output)

                        sample_pose = pose(sample_output)
                        prev_pose = sample_pose.detach()
                        prev_file_id = current_file_id

                    first = batch_outputs[0]
                    if isinstance(first, dict):
                        output = {
                            key: torch.cat([item[key] for item in batch_outputs if isinstance(item, dict)], dim=0)
                            for key in first
                        }
                    else:
                        output = torch.cat([item for item in batch_outputs if isinstance(item, torch.Tensor)], dim=0)
                else:
                    synchronize()
                    forward_started = time.perf_counter()
                    output = model(inputs)
                    synchronize()
                    forward_ms.append((time.perf_counter() - forward_started) * 1000.0)
                target_pose = targets["pose"] if isinstance(targets, dict) else targets
                outputs.append(pose(output).detach().cpu().numpy().astype(np.float32))
                targets_out.append(target_pose.detach().cpu().numpy().astype(np.float32))
        elapsed = time.perf_counter() - started
        return (
            np.concatenate(outputs, axis=0),
            np.concatenate(targets_out, axis=0),
            elapsed,
            np.asarray(forward_ms, dtype=np.float32),
        )

    def infer_int8_reference() -> tuple[np.ndarray, float, np.ndarray]:
        outputs: list[np.ndarray] = []
        forward_ms: list[float] = []
        started = time.perf_counter()
        for inputs, _targets in loader:
            cancel_check(cancel)
            x_np = inputs.numpy().astype(np.float32)
            forward_started = time.perf_counter()
            pose_np, _ = fast_cnn_int8_forward(x_np, int8_arrays, int8_metadata)
            forward_ms.append((time.perf_counter() - forward_started) * 1000.0)
            outputs.append(pose_np.astype(np.float32))
        elapsed = time.perf_counter() - started
        return (
            np.concatenate(outputs, axis=0),
            elapsed,
            np.asarray(forward_ms, dtype=np.float32),
        )

    float_pose, target_pose, float_elapsed, float_forward_ms = infer_model(float_model)
    cancel_check(cancel)
    int8_pose, int8_elapsed, int8_forward_ms = infer_int8_reference()
    cancel_check(cancel)
    int8_codes = np.round(int8_pose / scale).clip(-127, 127).astype(np.int8)
    return (
        float_pose,
        int8_pose,
        int8_codes,
        target_pose,
        trigger_seq,
        float_elapsed,
        int8_elapsed,
        float_forward_ms,
        int8_forward_ms,
        scale,
    )


def parse_pose_line(line: str) -> np.ndarray | None:
    text = line.strip()
    if "[" in text and "]" in text:
        text = text[text.find("[") + 1 : text.rfind("]")]
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 24:
        return None
    try:
        return np.asarray([float(part) for part in parts], dtype=np.float32)
    except ValueError:
        return None


def parse_timing(line: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?)", line):
        try:
            result[key] = float(value)
        except ValueError:
            pass
    return result


REMOTE_REPLAY_HELPER = r'''
import argparse
import os
import pty
import select
import struct
import subprocess
import sys
import threading
import time


SERIAL_HEADER = struct.Struct("<IBBHII")


def split_serial_frames(path):
    data = open(path, "rb").read()
    frames = []
    cursor = 0
    while cursor + SERIAL_HEADER.size <= len(data):
        _magic, _version, _frame_type, payload_len, _frame_seq, _checksum = SERIAL_HEADER.unpack_from(data, cursor)
        end = cursor + SERIAL_HEADER.size + payload_len
        if end > len(data):
            break
        frames.append(data[cursor:end])
        cursor = end
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--weights", default="pl_accel_v6_weights.bin")
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--replay-bin", required=True)
    parser.add_argument("--max-infer", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--write-delay-ms", type=float, default=0.0)
    parser.add_argument("--idle-timeout-sec", type=float, default=30.0)
    args = parser.parse_args()

    frames = split_serial_frames(args.replay_bin)
    frame_index = 0
    pose_count = 0
    replay_started = False
    timed_out = False
    writer_stop = threading.Event()
    writer_thread = None
    frame_lock = threading.Lock()

    master_fd, slave_fd = pty.openpty()
    os.set_blocking(master_fd, False)
    slave_name = os.ttyname(slave_fd)
    stdout_master_fd, stdout_slave_fd = pty.openpty()
    os.set_blocking(stdout_master_fd, False)
    cmd = [
        args.runner,
        "--tty",
        slave_name,
        "--no-esp-config",
        "--print-every",
        "1",
        "--max-infer",
        str(args.max_infer),
        "--weights",
        args.weights,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=args.remote_dir,
        stdout=stdout_slave_fd,
        stderr=stdout_slave_fd,
        close_fds=True,
        bufsize=0,
    )
    os.close(stdout_slave_fd)

    def drain_master():
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0)
            if not ready:
                return
            try:
                data = os.read(master_fd, 4096)
            except BlockingIOError:
                return
            except OSError:
                return
            if not data:
                return

    def write_until(target_frame_count):
        nonlocal frame_index
        wrote = 0
        target_frame_count = min(target_frame_count, len(frames))
        while not writer_stop.is_set():
            with frame_lock:
                if frame_index >= target_frame_count:
                    break
                frame = frames[frame_index]
                frame_index += 1
            drain_master()
            while not writer_stop.is_set():
                try:
                    os.write(master_fd, frame)
                    break
                except BlockingIOError:
                    drain_master()
                    time.sleep(0.001)
                except OSError:
                    return wrote
            wrote += 1
            if args.write_delay_ms > 0:
                time.sleep(args.write_delay_ms / 1000.0)
        drain_master()
        with frame_lock:
            current_frame_index = frame_index
        sys.stdout.write(f"replay_progress frames={current_frame_index}/{len(frames)} poses={pose_count}/{args.max_infer}\n")
        sys.stdout.flush()
        return wrote

    def frames_needed_for_pose_count(pose_target):
        return args.window_size + max(0, pose_target - 1) * args.window_stride

    def start_replay():
        nonlocal replay_started, writer_thread
        if replay_started:
            return
        replay_started = True
        # setup_tty_raw() and its tcflush() happen before the runner prints
        # "ESP serial:", so frames sent from this point will not be discarded
        # by the runner's startup flush. The PTY buffers them until the RX
        # thread starts reading.
        target_frames = frames_needed_for_pose_count(args.max_infer)
        writer_thread = threading.Thread(target=write_until, args=(target_frames,), daemon=True)
        writer_thread.start()

    last_activity = time.monotonic()
    stdout_buffer = ""

    while True:
        drain_master()
        if proc.poll() is not None:
            break
        ready, _, _ = select.select([stdout_master_fd], [], [], 1.0)
        if not ready:
            if time.monotonic() - last_activity > args.idle_timeout_sec:
                sys.stdout.write(
                    f"replay_timeout frames={frame_index}/{len(frames)} poses={pose_count}/{args.max_infer}\n"
                )
                sys.stdout.flush()
                timed_out = True
                proc.terminate()
                break
            continue
        try:
            chunk = os.read(stdout_master_fd, 4096)
        except BlockingIOError:
            continue
        except OSError:
            break
        if not chunk:
            break
        last_activity = time.monotonic()
        stdout_buffer += chunk.decode(errors="replace")
        while "\n" in stdout_buffer:
            line, stdout_buffer = stdout_buffer.split("\n", 1)
            line = line.rstrip("\r") + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()
            if not replay_started and (
                line.startswith("ESP serial:")
                or "threads started" in line
            ):
                start_replay()
            if line.lstrip().startswith("pose trigger="):
                pose_count += 1
                if pose_count >= args.max_infer:
                    break
        if pose_count >= args.max_infer:
            break

    writer_stop.set()
    if writer_thread is not None:
        writer_thread.join(timeout=2.0)
    try:
        os.close(master_fd)
    except OSError:
        pass
    try:
        os.close(stdout_master_fd)
    except OSError:
        pass
    rc = proc.wait()
    try:
        os.close(slave_fd)
    except OSError:
        pass
    if timed_out or pose_count < args.max_infer:
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
'''


def ssh_base(user: str, host: str, port: int) -> list[str]:
    return [
        "ssh",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{user}@{host}",
    ]


def remote_write_bytes(
    *,
    user: str,
    host: str,
    port: int,
    remote_path: str,
    payload: bytes,
    cancel: CancellationToken | None = None,
) -> None:
    command = ssh_base(user, host, port) + [f"cat > {shlex.quote(remote_path)}"]
    run_subprocess(command, input=payload, check=True, cancel=cancel)


def remote_write_text(
    *,
    user: str,
    host: str,
    port: int,
    remote_path: str,
    text: str,
    cancel: CancellationToken | None = None,
) -> None:
    command = ssh_base(user, host, port) + [f"cat > {shlex.quote(remote_path)}"]
    run_subprocess(command, input=text, text=True, check=True, cancel=cancel)


def remote_read_bytes(
    *,
    user: str,
    host: str,
    port: int,
    remote_path: str,
    cancel: CancellationToken | None = None,
) -> bytes:
    command = ssh_base(user, host, port) + [f"cat {shlex.quote(remote_path)}"]
    result = run_subprocess(command, capture_output=True, check=True, cancel=cancel)
    return result.stdout


def run_remote(
    *,
    user: str,
    host: str,
    port: int,
    command: str,
    cancel: CancellationToken | None = None,
) -> str:
    result = run_subprocess(
        ssh_base(user, host, port) + [command],
        capture_output=True,
        text=True,
        check=True,
        cancel=cancel,
    )
    return result.stdout + result.stderr


def run_ps_c_inference(
    *,
    ml_root: Path,
    config_path: Path,
    cache_dir: Path,
    local_output_dir: Path,
    windows: int,
    user: str,
    host: str,
    port: int,
    ps_remote_dir: str,
    status: Callable[[str], None] | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def emit(text: str) -> None:
        if status is not None:
            status(text)

    from export_ps_model import resolve, write_ps_inputs, write_ps_model

    config = load_json(config_path)
    checkpoint_path = resolve(config_path.parent, config["checkpoint"])
    cancel_check(cancel)
    output_dir = ensure_int8_export(ml_root, config_path, cancel=cancel)

    local_output_dir.mkdir(parents=True, exist_ok=True)
    ps_model = local_output_dir / "ps_model.bin"
    ps_input = local_output_dir / "ps_input.bin"
    emit("exporting PS C model/input binaries")
    count = write_ps_inputs(
        ml_root=ml_root,
        config_path=config_path,
        cache_dir=cache_dir,
        windows=windows,
        output_path=ps_input,
    )
    write_ps_model(
        ml_root=ml_root,
        checkpoint_path=checkpoint_path,
        int8_weights_path=output_dir / "weights_int8.npz",
        model_json_path=output_dir / "model_int8.json",
        output_path=ps_model,
    )

    c_source = Path(__file__).with_name("ps_pose_infer.c").read_text(encoding="utf-8")
    run_remote(user=user, host=host, port=port, command=f"mkdir -p {shlex.quote(ps_remote_dir)}", cancel=cancel)
    emit("uploading PS C runner/model/input")
    remote_write_text(user=user, host=host, port=port, remote_path=f"{ps_remote_dir}/ps_pose_infer.c", text=c_source, cancel=cancel)
    remote_write_bytes(user=user, host=host, port=port, remote_path=f"{ps_remote_dir}/ps_model.bin", payload=ps_model.read_bytes(), cancel=cancel)
    remote_write_bytes(user=user, host=host, port=port, remote_path=f"{ps_remote_dir}/ps_input.bin", payload=ps_input.read_bytes(), cancel=cancel)

    emit("compiling PS C runner on board")
    compile_cmd = f"cd {shlex.quote(ps_remote_dir)} && gcc -O3 -o ps_pose_infer ps_pose_infer.c -lm"
    run_remote(user=user, host=host, port=port, command=compile_cmd, cancel=cancel)

    def run_mode(mode: str) -> tuple[np.ndarray, np.ndarray]:
        emit(f"running PS {mode} C inference")
        cmd = (
            f"cd {shlex.quote(ps_remote_dir)} && "
            f"./ps_pose_infer --mode {mode} --model ps_model.bin --input ps_input.bin "
            f"--output ps_{mode}_pose.bin --timing ps_{mode}_timing.bin"
        )
        log = run_remote(user=user, host=host, port=port, command=cmd, cancel=cancel)
        emit(log.strip().splitlines()[-1] if log.strip() else f"PS {mode} done")
        pose_bytes = remote_read_bytes(user=user, host=host, port=port, remote_path=f"{ps_remote_dir}/ps_{mode}_pose.bin", cancel=cancel)
        timing_bytes = remote_read_bytes(user=user, host=host, port=port, remote_path=f"{ps_remote_dir}/ps_{mode}_timing.bin", cancel=cancel)
        poses = np.frombuffer(pose_bytes, dtype=np.float32).reshape((-1, POSE_DIM)).copy()
        timing = np.frombuffer(timing_bytes, dtype=np.float32).copy()
        if len(poses) != count:
            raise RuntimeError(f"PS {mode} produced {len(poses)}/{count} poses")
        return poses, timing

    ps_float32_pose, ps_float32_ms = run_mode("float32")
    ps_int8_pose, ps_int8_ms = run_mode("int8")
    return ps_float32_pose, ps_int8_pose, ps_float32_ms, ps_int8_ms


def run_fpga_inference(
    *,
    frames: bytes,
    expected_outputs: int,
    user: str,
    host: str,
    port: int,
    remote_dir: str,
    runner: str,
    weight_path: Path,
    window_size: int,
    window_stride: int,
    write_delay_ms: float,
    status: Callable[[str], None] | None = None,
    cancel: CancellationToken | None = None,
) -> tuple[np.ndarray, list[dict[str, float]], float, str]:
    def emit(text: str) -> None:
        if status is not None:
            status(text)

    stamp = int(time.time() * 1000)
    replay_path = f"/tmp/fpga_csv_replay_{stamp}.bin"
    helper_path = f"/tmp/fpga_csv_replay_runner_{stamp}.py"
    remote_weight_name = "pl_accel_v6_weights.bin"
    remote_weight_path = f"{remote_dir.rstrip('/')}/{remote_weight_name}"
    cancel_check(cancel)
    emit(f"uploading PL weights ({weight_path.stat().st_size / 1024:.1f} KiB)")
    remote_write_bytes(user=user, host=host, port=port, remote_path=remote_weight_path, payload=weight_path.read_bytes(), cancel=cancel)
    emit(f"uploading FPGA replay binary ({len(frames) / (1024 * 1024):.1f} MiB)")
    remote_write_bytes(user=user, host=host, port=port, remote_path=replay_path, payload=frames, cancel=cancel)
    emit("uploading FPGA replay helper")
    remote_write_text(user=user, host=host, port=port, remote_path=helper_path, text=REMOTE_REPLAY_HELPER, cancel=cancel)

    remote_cmd = (
        f"python3 {shlex.quote(helper_path)} "
        f"--runner {shlex.quote(runner)} "
        f"--weights {shlex.quote(remote_weight_name)} "
        f"--remote-dir {shlex.quote(remote_dir)} "
        f"--replay-bin {shlex.quote(replay_path)} "
        f"--max-infer {expected_outputs} "
        f"--window-size {window_size} "
        f"--window-stride {window_stride} "
        f"--write-delay-ms {write_delay_ms}"
    )
    command = ssh_base(user, host, port) + [remote_cmd]
    poses: list[np.ndarray] = []
    timings: list[dict[str, float]] = []
    output_lines: list[str] = []
    started = time.perf_counter()
    proc: subprocess.Popen[Any] | None = None
    emit(f"starting remote runner; FPGA outputs 0/{expected_outputs}")
    try:
        proc = start_subprocess(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cancel=cancel,
        )
        if proc.stdout is None:
            raise RuntimeError("ssh stdout unavailable")
        last_update = time.perf_counter()
        for line in proc.stdout:
            cancel_check(cancel)
            output_lines.append(line)
            pose = parse_pose_line(line)
            if pose is not None:
                poses.append(pose)
                timing = parse_timing(line)
                timings.append(timing)
                hls_ms = timing.get("hls_ms")
                gap_ms = timing.get("gap_ms")
                esp_gap_ms = timing.get("esp_gap_ms")
                esp_drop_count = timing.get("esp_drop_count")
                extra = ""
                if hls_ms is not None:
                    extra += f" hls={hls_ms:.3f}ms"
                if esp_gap_ms is not None and esp_gap_ms >= 0:
                    extra += f" esp_gap={esp_gap_ms:.3f}ms"
                if esp_drop_count is not None and esp_drop_count > 0:
                    extra += f" esp_drop={int(esp_drop_count)}"
                if gap_ms is not None and gap_ms >= 0:
                    extra += f" pl_gap={gap_ms:.3f}ms"
                emit(f"FPGA outputs {len(poses)}/{expected_outputs}{extra}")
                continue
            now = time.perf_counter()
            text = line.strip()
            if text and now - last_update >= 0.5:
                emit(f"FPGA runner: {text[:120]}")
                last_update = now
        rc = proc.wait()
        elapsed = time.perf_counter() - started
        cancel_check(cancel)
    finally:
        if proc is not None and cancel is not None:
            cancel.unregister(proc)
        emit("cleaning remote replay files")
        if cancel is not None and cancel.is_cancelled():
            runner_pattern = shlex.quote(pkill_no_self_pattern(Path(runner).name or runner))
            helper_pattern = shlex.quote(pkill_no_self_pattern(helper_path))
            cleanup_cmd = (
                f"pkill -f {runner_pattern} 2>/dev/null || true; "
                f"pkill -f {helper_pattern} 2>/dev/null || true; "
                f"rm -f {shlex.quote(replay_path)} {shlex.quote(helper_path)}"
            )
        else:
            cleanup_cmd = f"rm -f {shlex.quote(replay_path)} {shlex.quote(helper_path)}"
        try:
            subprocess.run(
                ssh_base(user, host, port) + [cleanup_cmd],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
        except Exception:  # noqa: BLE001
            pass

    stdout_text = "".join(output_lines)
    if not poses:
        raise RuntimeError(
            "FPGA produced no pose outputs. "
            "The replay helper starts sending frames after the runner opens its PTY; "
            "check whether the remote log reaches 'threads started' and 'replay_progress'.\n\n"
            + (stdout_text.strip() or f"remote runner failed: {rc}")
        )
    if rc != 0 or len(poses) < expected_outputs:
        raise RuntimeError(
            f"FPGA produced only {len(poses)}/{expected_outputs} pose outputs. "
            "Comparison was not saved because full CSV inference did not complete.\n\n"
            + stdout_text.strip()
        )
    return np.stack(poses).astype(np.float32) if poses else np.zeros((0, 24), dtype=np.float32), timings, elapsed, stdout_text


def mae(a: np.ndarray, b: np.ndarray) -> float:
    count = min(len(a), len(b))
    if count <= 0:
        return 0.0
    return float(np.abs(a[:count] - b[:count]).mean())


def save_result(result: RunResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{result.csv_path.stem}_{result.model_name}_{stamp}"
    npz_path = output_dir / f"{stem}.npz"
    summary_path = output_dir / f"{stem}.json"
    count = result.sample_count

    timing_keys = sorted({key for item in result.fpga_timing for key in item})
    timing_array = np.zeros((len(result.fpga_timing), len(timing_keys)), dtype=np.float32)
    for row_index, item in enumerate(result.fpga_timing):
        for col_index, key in enumerate(timing_keys):
            timing_array[row_index, col_index] = float(item.get(key, np.nan))
    hls_values = np.asarray(
        [item["hls_ms"] for item in result.fpga_timing if "hls_ms" in item],
        dtype=np.float32,
    )
    esp_gap_values = np.asarray(
        [item["esp_gap_ms"] for item in result.fpga_timing if item.get("esp_gap_ms", -1.0) >= 0.0],
        dtype=np.float32,
    )
    esp_drop_values = np.asarray(
        [item["esp_drop_count"] for item in result.fpga_timing if "esp_drop_count" in item],
        dtype=np.float32,
    )

    save_payload = {
        "trigger_seq": result.trigger_seq,
        "target_pose": result.target_pose,
        "pc_float32_pose": result.pc_float32_pose,
        "pc_int8_pose": result.pc_int8_pose,
        "pc_int8_codes": result.pc_int8_codes,
        "fpga_pose": result.fpga_pose,
        "fpga_timing": timing_array,
        "fpga_timing_keys": np.asarray(timing_keys, dtype="U64"),
        "pc_float32_forward_ms": result.pc_float32_forward_ms,
        "pc_int8_forward_ms": result.pc_int8_forward_ms,
        "fpga_hls_ms": hls_values,
        "output_scale": np.asarray(result.output_scale, dtype=np.float32),
    }
    if result.ps_float32_pose is not None:
        save_payload["ps_float32_pose"] = result.ps_float32_pose
    if result.ps_int8_pose is not None:
        save_payload["ps_int8_pose"] = result.ps_int8_pose
    if result.ps_float32_forward_ms is not None:
        save_payload["ps_float32_forward_ms"] = result.ps_float32_forward_ms
    if result.ps_int8_forward_ms is not None:
        save_payload["ps_int8_forward_ms"] = result.ps_int8_forward_ms
    np.savez_compressed(npz_path, **save_payload)
    summary = {
        "csv": str(result.csv_path),
        "model_name": result.model_name,
        "samples": count,
        "output_npz": str(npz_path),
        "pc_float32_elapsed_sec": result.pc_float32_elapsed_sec,
        "pc_int8_elapsed_sec": result.pc_int8_elapsed_sec,
        "fpga_elapsed_sec": result.fpga_elapsed_sec,
        "pc_timing_mode": "realtime_single_window_batch1",
        "pc_float32_windows_per_sec": count / max(result.pc_float32_elapsed_sec, 1e-9),
        "pc_int8_windows_per_sec": count / max(result.pc_int8_elapsed_sec, 1e-9),
        "fpga_windows_per_sec": len(result.fpga_pose) / max(result.fpga_elapsed_sec, 1e-9),
        "pc_float32_forward_ms_mean": float(np.mean(result.pc_float32_forward_ms)) if len(result.pc_float32_forward_ms) else 0.0,
        "pc_float32_forward_ms_median": float(np.median(result.pc_float32_forward_ms)) if len(result.pc_float32_forward_ms) else 0.0,
        "pc_int8_forward_ms_mean": float(np.mean(result.pc_int8_forward_ms)) if len(result.pc_int8_forward_ms) else 0.0,
        "pc_int8_forward_ms_median": float(np.median(result.pc_int8_forward_ms)) if len(result.pc_int8_forward_ms) else 0.0,
        "fpga_hls_ms_mean": float(np.mean(hls_values)) if len(hls_values) else 0.0,
        "fpga_hls_ms_median": float(np.median(hls_values)) if len(hls_values) else 0.0,
        "esp_gap_ms_mean": float(np.mean(esp_gap_values)) if len(esp_gap_values) else None,
        "esp_gap_ms_median": float(np.median(esp_gap_values)) if len(esp_gap_values) else None,
        "esp_drop_count_total": int(np.sum(esp_drop_values)) if len(esp_drop_values) else 0,
        "esp_drop_count_max": int(np.max(esp_drop_values)) if len(esp_drop_values) else 0,
        "mae_fpga_vs_pc_float32": mae(result.fpga_pose, result.pc_float32_pose),
        "mae_fpga_vs_pc_int8": mae(result.fpga_pose, result.pc_int8_pose),
        "mae_ps_float32_vs_pc_float32": mae(result.ps_float32_pose, result.pc_float32_pose) if result.ps_float32_pose is not None else None,
        "mae_ps_int8_vs_pc_int8": mae(result.ps_int8_pose, result.pc_int8_pose) if result.ps_int8_pose is not None else None,
        "ps_float32_ms_mean": float(np.mean(result.ps_float32_forward_ms)) if result.ps_float32_forward_ms is not None and len(result.ps_float32_forward_ms) else None,
        "ps_int8_ms_mean": float(np.mean(result.ps_int8_forward_ms)) if result.ps_int8_forward_ms is not None and len(result.ps_int8_forward_ms) else None,
        "mae_pc_int8_vs_float32": mae(result.pc_int8_pose, result.pc_float32_pose),
        "mae_pc_float32_vs_target": mae(result.pc_float32_pose, result.target_pose),
        "mae_fpga_vs_target": mae(result.fpga_pose, result.target_pose),
        "output_scale": result.output_scale,
        "fpga_timing_keys": timing_keys,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return npz_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    ml_root = default_ml_root()
    this_tool = tool_root()
    parser = argparse.ArgumentParser(description="Compare CSV PC float32/INT8 inference with FPGA runner replay.")
    parser.add_argument("csv", nargs="?", type=Path, default=ml_root / "infer_csv" / "sync_csi_pose_001.csv")
    parser.add_argument("--pattern", default="sync_csi_pose*.csv", help="Used when csv argument is a directory.")
    parser.add_argument("--ml-root", type=Path, default=ml_root)
    parser.add_argument("--config", type=Path, default=ml_root / "configs" / "fast_cnn_int8.json")
    parser.add_argument("--cache-dir", type=Path, default=this_tool / "_csv_compare_cache")
    parser.add_argument("--output-dir", type=Path, default=this_tool / "outputs")
    parser.add_argument("--windows", type=int, default=0, help="0 means all valid windows.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Deprecated for PC timing. PC inference is measured as realtime batch=1.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--node-count", type=int, default=3)
    parser.add_argument("--pair-count", type=int, default=0)
    parser.add_argument("--subcarrier-remap", default="esp32_htltf_ht40_above_nonstbc")
    parser.add_argument("--host", default="192.168.1.15")
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--remote-dir", default="/home/root")
    parser.add_argument("--ps-remote-dir", default="/home/root/ps_infer")
    parser.add_argument("--skip-ps", action="store_true")
    parser.add_argument("--runner", default="./esp_pose_pl_runner_rt")
    parser.add_argument("--weights", type=Path, default=this_tool / "pl_accel_v6_weights.bin")
    parser.add_argument(
        "--write-delay-ms",
        type=float,
        default=3.2,
        help="Delay between replay frames. Keep >=3.2ms for pl_accel_v6 to avoid dropping ping-pong input buffers.",
    )
    parser.add_argument("--no-gui", action="store_true")
    return parser


def run_compare(
    args: argparse.Namespace,
    status: Callable[[str], None] | None = None,
    cancel: CancellationToken | None = None,
) -> RunResult:
    def emit(text: str) -> None:
        if status is not None:
            status(text)

    csv_path = args.csv.resolve()
    input_dir, input_pattern, csv_files = resolve_input(csv_path, args.pattern)
    ml_root = args.ml_root.resolve()
    config_path = args.config.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    config = load_json(config_path)
    model_name = Path(str(config.get("checkpoint", ""))).parent.name or config_path.stem
    cancel_check(cancel)

    if cache_is_ready(cache_dir, csv_path, args.pattern):
        emit("using existing CSV cache")
    else:
        emit("preparing CSV cache")
        prepare_csv_cache(
            ml_root=ml_root,
            input_path=csv_path,
            pattern=args.pattern,
            cache_dir=cache_dir,
            node_count=args.node_count,
            pair_count=args.pair_count,
            subcarrier_remap=args.subcarrier_remap,
        )
        emit("CSV cache ready")

    cancel_check(cancel)
    emit("running PC float32 and INT8 inference (realtime batch=1)")
    (
        pc_float32_pose,
        pc_int8_pose,
        pc_int8_codes,
        target_pose,
        trigger_seq,
        pc_float_elapsed,
        pc_int8_elapsed,
        pc_float_forward_ms,
        pc_int8_forward_ms,
        output_scale,
    ) = run_pc_inference(
        ml_root=ml_root,
        config_path=config_path,
        cache_dir=cache_dir,
        windows=args.windows,
        batch_size=args.batch_size,
        device_text=args.device,
        cancel=cancel,
    )
    cancel_check(cancel)
    emit(
        "PC inference done "
        f"(total float32 {pc_float_elapsed:.3f}s, int8 {pc_int8_elapsed:.3f}s; "
        f"forward avg float32 {float(np.mean(pc_float_forward_ms)):.3f}ms, "
        f"int8 {float(np.mean(pc_int8_forward_ms)):.3f}ms)"
    )

    expected = len(pc_float32_pose)
    window_size = int(config.get("window_size", 10))
    window_stride = int(config.get("window_stride", 10))

    ps_float32_pose = None
    ps_int8_pose = None
    ps_float32_ms = None
    ps_int8_ms = None
    if not args.skip_ps:
        ps_float32_pose, ps_int8_pose, ps_float32_ms, ps_int8_ms = run_ps_c_inference(
            ml_root=ml_root,
            config_path=config_path,
            cache_dir=cache_dir,
            local_output_dir=output_dir / "_ps_export",
            windows=expected,
            user=args.user,
            host=args.host,
            port=args.port,
            ps_remote_dir=args.ps_remote_dir,
            status=emit,
            cancel=cancel,
        )

    cancel_check(cancel)
    emit(f"building USB-compatible serial replay frames for {len(csv_files)} CSV file(s)")
    file_counts = selected_file_window_counts(
        ml_root=ml_root,
        config_path=config_path,
        cache_dir=cache_dir,
        windows=args.windows,
    )
    fpga_pose_parts: list[np.ndarray] = []
    fpga_timing: list[dict[str, float]] = []
    fpga_elapsed = 0.0
    for file_id, file_path in enumerate(csv_files):
        cancel_check(cancel)
        expected_for_file = int(file_counts.get(file_id, 0))
        if expected_for_file <= 0:
            continue
        emit(f"running FPGA replay file {file_id + 1}/{len(csv_files)}: {file_path.name} windows={expected_for_file}")
        frames, _ = csv_to_serial_frames(file_path, node_count=args.node_count)
        pose_part, timing_part, elapsed_part, _ = run_fpga_inference(
            frames=frames,
            expected_outputs=expected_for_file,
            user=args.user,
            host=args.host,
            port=args.port,
            remote_dir=args.remote_dir,
            runner=args.runner,
            weight_path=args.weights.resolve(),
            window_size=window_size,
            window_stride=window_stride,
            write_delay_ms=args.write_delay_ms,
            status=emit,
            cancel=cancel,
        )
        fpga_pose_parts.append(pose_part)
        fpga_timing.extend(timing_part)
        fpga_elapsed += elapsed_part
    fpga_pose = np.concatenate(fpga_pose_parts, axis=0) if fpga_pose_parts else np.zeros((0, POSE_DIM), dtype=np.float32)

    cancel_check(cancel)
    result = RunResult(
        csv_path=csv_path,
        model_name=model_name,
        trigger_seq=trigger_seq,
        target_pose=target_pose,
        pc_float32_pose=pc_float32_pose,
        pc_int8_pose=pc_int8_pose,
        pc_int8_codes=pc_int8_codes,
        fpga_pose=fpga_pose,
        fpga_timing=fpga_timing,
        ps_float32_pose=ps_float32_pose,
        ps_int8_pose=ps_int8_pose,
        ps_float32_forward_ms=ps_float32_ms,
        ps_int8_forward_ms=ps_int8_ms,
        pc_float32_elapsed_sec=pc_float_elapsed,
        pc_int8_elapsed_sec=pc_int8_elapsed,
        pc_float32_forward_ms=pc_float_forward_ms,
        pc_int8_forward_ms=pc_int8_forward_ms,
        fpga_elapsed_sec=fpga_elapsed,
        output_scale=output_scale,
    )
    emit("saving comparison outputs")
    result.output_npz, result.summary_json = save_result(result, output_dir)
    emit(f"saved outputs: {result.output_npz}")
    return result


class CompareApp(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.result: RunResult | None = None
        self.cancel_token: CancellationToken | None = None
        self.worker_thread: threading.Thread | None = None
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.index_var = tk.IntVar(value=0)
        self.show_target_var = tk.BooleanVar(value=True)
        self.show_pc_float32_var = tk.BooleanVar(value=True)
        self.show_pc_int8_var = tk.BooleanVar(value=True)
        self.show_ps_float32_var = tk.BooleanVar(value=True)
        self.show_ps_int8_var = tk.BooleanVar(value=True)
        self.show_fpga_pl_var = tk.BooleanVar(value=True)
        self.title("CSV PC vs FPGA Pose Compare")
        self.geometry("1320x860")
        self.minsize(1080, 720)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_window_default()
        self.after(50, self._poll)

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="CSV").grid(row=0, column=0, sticky="w")
        self.csv_var = tk.StringVar(value=str(self.args.csv))
        ttk.Entry(top, textvariable=self.csv_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="File", command=self._browse_csv).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(top, text="Folder", command=self._browse_folder).grid(row=0, column=3, padx=(0, 6))
        self.run_button = ttk.Button(top, text="Run", command=self._start)
        self.run_button.grid(row=0, column=4, padx=(0, 6))
        self.stop_button = ttk.Button(top, text="Stop", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=5)

        opts = ttk.Frame(self, padding=(10, 0, 10, 8))
        opts.grid(row=1, column=0, sticky="ew")
        self.host_var = tk.StringVar(value=self.args.host)
        self.runner_var = tk.StringVar(value=self.args.runner)
        self.windows_var = tk.StringVar(value=str(self.args.windows) if self.args.windows > 0 else "calculating")
        ttk.Label(opts, text="Host").grid(row=0, column=0)
        ttk.Entry(opts, textvariable=self.host_var, width=16).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(opts, text="Runner").grid(row=0, column=2)
        ttk.Entry(opts, textvariable=self.runner_var, width=24).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(opts, text="Windows").grid(row=0, column=4)
        ttk.Entry(opts, textvariable=self.windows_var, width=8).grid(row=0, column=5, padx=(4, 12))
        self.status_var = tk.StringVar(value="ready")
        ttk.Label(opts, textvariable=self.status_var).grid(row=0, column=6, sticky="w")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        left = ttk.Frame(body)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        body.add(left, weight=3)
        right = ttk.Frame(body)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)
        body.add(right, weight=2)

        self.summary_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.summary_var).grid(row=0, column=0, sticky="ew")
        self.canvas = tk.Canvas(left, background="#0d1117", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.scale = ttk.Scale(left, from_=0, to=0, orient="horizontal", variable=self.index_var, command=lambda _v: self._draw())
        self.scale.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.sample_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.sample_var, justify="left").grid(row=0, column=0, sticky="ew")
        toggles = ttk.Frame(right)
        toggles.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(toggles, text="target", variable=self.show_target_var, command=self._draw).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(toggles, text="pc f32", variable=self.show_pc_float32_var, command=self._draw).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(toggles, text="pc int8", variable=self.show_pc_int8_var, command=self._draw).grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Checkbutton(toggles, text="ps f32", variable=self.show_ps_float32_var, command=self._draw).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(toggles, text="ps int8", variable=self.show_ps_int8_var, command=self._draw).grid(row=1, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(toggles, text="pl int8", variable=self.show_fpga_pl_var, command=self._draw).grid(row=1, column=2, sticky="w", padx=(8, 0))
        columns = ("metric", "value")
        self.table = ttk.Treeview(right, columns=columns, show="headings", height=10)
        self.table.heading("metric", text="Metric")
        self.table.heading("value", text="Value")
        self.table.column("metric", width=210)
        self.table.column("value", width=210)
        self.table.grid(row=2, column=0, sticky="nsew", pady=(8, 0))

    def _browse_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_var.set(path)
            self.args.csv = Path(path)
            self._refresh_window_default()

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.csv_var.set(path)
            self.args.csv = Path(path)
            self._refresh_window_default()

    def _refresh_window_default(self) -> None:
        self.windows_var.set("calculating")
        self.status_var.set("preparing CSV to calculate max windows")
        self.run_button.configure(state="disabled")
        self.args.csv = Path(self.csv_var.get())
        threading.Thread(target=self._window_count_worker, daemon=True).start()

    def _is_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _window_count_worker(self) -> None:
        try:
            count = prepare_and_count_windows(self.args)
            self.queue.put(("window_count", count))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("window_count_error", str(exc)))

    def _start(self) -> None:
        if self._is_running():
            return
        self.args.csv = Path(self.csv_var.get())
        self.args.host = self.host_var.get()
        self.args.runner = self.runner_var.get()
        try:
            self.args.windows = int(self.windows_var.get())
        except ValueError:
            messagebox.showerror(self.title(), "Windows must be an integer.")
            return
        self.cancel_token = CancellationToken()
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("starting")
        self.worker_thread = threading.Thread(target=self._worker, args=(self.cancel_token,), daemon=True)
        self.worker_thread.start()

    def _stop(self) -> None:
        if self.cancel_token is None:
            return
        self.status_var.set("stopping")
        self.stop_button.configure(state="disabled")
        self.cancel_token.cancel()

    def _on_close(self) -> None:
        if self.cancel_token is not None:
            self.cancel_token.cancel()
        self.destroy()

    def _worker(self, cancel: CancellationToken) -> None:
        try:
            result = run_compare(self.args, status=lambda text: self.queue.put(("status", text)), cancel=cancel)
            self.queue.put(("result", result))
        except RunCancelled as exc:
            self.queue.put(("cancelled", str(exc)))
        except Exception as exc:  # noqa: BLE001
            self.queue.put(("error", str(exc)))

    def _poll(self) -> None:
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(str(payload))
            elif kind == "window_count":
                count = int(payload)
                self.args.windows = count
                self.windows_var.set(str(count))
                if not self._is_running():
                    self.status_var.set(f"ready: max windows={count}")
                    self.run_button.configure(state="normal")
            elif kind == "window_count_error":
                self.windows_var.set("0")
                if not self._is_running():
                    self.status_var.set("failed to calculate max windows")
                    self.run_button.configure(state="normal")
                    messagebox.showerror(self.title(), f"Could not calculate max windows:\n{payload}")
            elif kind == "result":
                self.result = payload if isinstance(payload, RunResult) else None
                self._load_result()
                self.cancel_token = None
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
            elif kind == "cancelled":
                self.status_var.set("stopped")
                self.cancel_token = None
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
            elif kind == "error":
                self.status_var.set("error")
                self.cancel_token = None
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                messagebox.showerror(self.title(), str(payload))
        self.after(50, self._poll)

    def _load_result(self) -> None:
        if self.result is None:
            return
        count = max(self.result.sample_count - 1, 0)
        self.scale.configure(to=count)
        self.index_var.set(0)
        self.status_var.set("done")
        pc_float_fps = self.result.sample_count / max(self.result.pc_float32_elapsed_sec, 1e-9)
        pc_int8_fps = self.result.sample_count / max(self.result.pc_int8_elapsed_sec, 1e-9)
        fpga_fps = len(self.result.fpga_pose) / max(self.result.fpga_elapsed_sec, 1e-9)
        hls_values = np.asarray([item["hls_ms"] for item in self.result.fpga_timing if "hls_ms" in item], dtype=np.float32)
        esp_gap_values = np.asarray(
            [item["esp_gap_ms"] for item in self.result.fpga_timing if item.get("esp_gap_ms", -1.0) >= 0.0],
            dtype=np.float32,
        )
        esp_drop_values = np.asarray(
            [item["esp_drop_count"] for item in self.result.fpga_timing if "esp_drop_count" in item],
            dtype=np.float32,
        )
        esp_text = ""
        if len(esp_gap_values):
            esp_text += f" esp_gap={float(np.mean(esp_gap_values)):.3f}ms"
        if len(esp_drop_values):
            esp_text += f" esp_drop_total={int(np.sum(esp_drop_values))}"
        self.summary_var.set(
            f"samples={self.result.sample_count}  "
            f"PC realtime float32={pc_float_fps:.2f} win/s  "
            f"PC realtime int8={pc_int8_fps:.2f} win/s  "
            f"FPGA pipeline={fpga_fps:.2f} win/s  "
            f"pure ms: f32={float(np.mean(self.result.pc_float32_forward_ms)):.3f} "
            f"i8={float(np.mean(self.result.pc_int8_forward_ms)):.3f} "
            f"hls={float(np.mean(hls_values)) if len(hls_values) else 0.0:.3f}  "
            f"{esp_text}  "
            f"saved={self.result.output_npz}"
        )
        self._fill_table()
        self._draw()

    def _fill_table(self) -> None:
        self.table.delete(*self.table.get_children())
        if self.result is None:
            return
        rows = [
            ("MAE FPGA vs PC float32", f"{mae(self.result.fpga_pose, self.result.pc_float32_pose):.6f}"),
            ("MAE FPGA vs PC int8", f"{mae(self.result.fpga_pose, self.result.pc_int8_pose):.6f}"),
            ("MAE PC int8 vs float32", f"{mae(self.result.pc_int8_pose, self.result.pc_float32_pose):.6f}"),
            ("MAE PC float32 vs target", f"{mae(self.result.pc_float32_pose, self.result.target_pose):.6f}"),
            ("MAE FPGA vs target", f"{mae(self.result.fpga_pose, self.result.target_pose):.6f}"),
            ("Output scale", f"{self.result.output_scale:.9g}"),
            ("PC timing mode", "realtime batch=1"),
            ("PC float32 realtime elapsed", f"{self.result.pc_float32_elapsed_sec:.3f} sec"),
            ("PC int8 realtime elapsed", f"{self.result.pc_int8_elapsed_sec:.3f} sec"),
            ("FPGA pipeline elapsed", f"{self.result.fpga_elapsed_sec:.3f} sec"),
            ("PC float32 forward mean", f"{float(np.mean(self.result.pc_float32_forward_ms)):.3f} ms"),
            ("PC float32 forward median", f"{float(np.median(self.result.pc_float32_forward_ms)):.3f} ms"),
            ("PC int8 forward mean", f"{float(np.mean(self.result.pc_int8_forward_ms)):.3f} ms"),
            ("PC int8 forward median", f"{float(np.median(self.result.pc_int8_forward_ms)):.3f} ms"),
        ]
        if self.result.ps_float32_pose is not None:
            rows.append(("MAE PS float32 vs PC float32", f"{mae(self.result.ps_float32_pose, self.result.pc_float32_pose):.6f}"))
        if self.result.ps_int8_pose is not None:
            rows.append(("MAE PS int8 vs PC int8", f"{mae(self.result.ps_int8_pose, self.result.pc_int8_pose):.6f}"))
        if self.result.ps_float32_forward_ms is not None and len(self.result.ps_float32_forward_ms):
            rows.append(("PS float32 mean", f"{float(np.mean(self.result.ps_float32_forward_ms)):.3f} ms"))
        if self.result.ps_int8_forward_ms is not None and len(self.result.ps_int8_forward_ms):
            rows.append(("PS int8 mean", f"{float(np.mean(self.result.ps_int8_forward_ms)):.3f} ms"))
        hls_values = np.asarray([item["hls_ms"] for item in self.result.fpga_timing if "hls_ms" in item], dtype=np.float32)
        if len(hls_values):
            rows.extend(
                [
                    ("FPGA HLS mean", f"{float(np.mean(hls_values)):.3f} ms"),
                    ("FPGA HLS median", f"{float(np.median(hls_values)):.3f} ms"),
                ]
            )
        esp_gap_values = np.asarray(
            [item["esp_gap_ms"] for item in self.result.fpga_timing if item.get("esp_gap_ms", -1.0) >= 0.0],
            dtype=np.float32,
        )
        esp_drop_values = np.asarray(
            [item["esp_drop_count"] for item in self.result.fpga_timing if "esp_drop_count" in item],
            dtype=np.float32,
        )
        esp_max_gap_values = np.asarray(
            [item["esp_max_gap"] for item in self.result.fpga_timing if "esp_max_gap" in item],
            dtype=np.float32,
        )
        if len(esp_gap_values):
            rows.append(("ESP input window gap mean", f"{float(np.mean(esp_gap_values)):.3f} ms"))
            rows.append(("ESP input window gap median", f"{float(np.median(esp_gap_values)):.3f} ms"))
        if len(esp_drop_values):
            rows.append(("ESP dropped CSI samples total", f"{int(np.sum(esp_drop_values))}"))
            rows.append(("ESP dropped CSI samples max/window", f"{int(np.max(esp_drop_values))}"))
        if len(esp_max_gap_values):
            rows.append(("ESP max consecutive gap", f"{int(np.max(esp_max_gap_values))} cycles"))
        if self.result.output_npz is not None:
            rows.append(("Saved NPZ", str(self.result.output_npz)))
        if self.result.summary_json is not None:
            rows.append(("Saved JSON", str(self.result.summary_json)))
        for row in rows:
            self.table.insert("", "end", values=row)

    def _pose_points(self, pose: np.ndarray, width: int, height: int) -> dict[str, tuple[float, float]]:
        margin = 54
        draw_w = max(width - 2 * margin, 1)
        draw_h = max(height - 2 * margin, 1)
        return {
            name: (margin + float(pose[i * 2]) * draw_w, margin + float(pose[i * 2 + 1]) * draw_h)
            for i, name in enumerate(JOINT_NAMES)
        }

    def _draw_pose(self, pose: np.ndarray, color: str, width: int, height: int, label: str, y: int) -> None:
        points = self._pose_points(pose, width, height)
        for a, b in SKELETON_EDGES:
            if a in points and b in points:
                self.canvas.create_line(*points[a], *points[b], fill=color, width=3)
        for x, yy in points.values():
            self.canvas.create_oval(x - 4, yy - 4, x + 4, yy + 4, fill=color, outline="")
        self.canvas.create_text(16, y, text=label, fill=color, anchor="nw", font=("Segoe UI", 11, "bold"))

    def _draw(self) -> None:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, height, fill="#0d1117", outline="")
        if self.result is None or self.result.sample_count <= 0:
            self.canvas.create_text(width / 2, height / 2, text="run comparison", fill="#d8dee9", font=("Segoe UI", 16))
            return
        idx = min(max(int(round(self.index_var.get())), 0), self.result.sample_count - 1)
        trig = int(self.result.trigger_seq[idx]) if idx < len(self.result.trigger_seq) else -1
        timing = self.result.fpga_timing[idx] if idx < len(self.result.fpga_timing) else {}
        self.sample_var.set(
            f"sample {idx + 1}/{self.result.sample_count}\n"
            f"trigger_seq={trig}\n"
            f"fpga hls_ms={timing.get('hls_ms', 0.0):.3f}  "
            f"esp_gap_ms={timing.get('esp_gap_ms', -1.0):.3f}  "
            f"esp_drop={int(timing.get('esp_drop_count', 0.0))}  "
            f"esp_max_gap={int(timing.get('esp_max_gap', 0.0))}\n"
            f"pl_gap_ms={timing.get('gap_ms', -1.0):.3f}  "
            f"total_ms={timing.get('total_ms', 0.0):.3f}"
        )
        legend_y = 14
        if self.show_target_var.get():
            self._draw_pose(self.result.target_pose[idx], "#9ca3af", width, height, "target", legend_y)
            legend_y += 20
        if self.show_pc_float32_var.get():
            self._draw_pose(self.result.pc_float32_pose[idx], "#22c55e", width, height, "pc float32", legend_y)
            legend_y += 20
        if self.show_pc_int8_var.get():
            self._draw_pose(self.result.pc_int8_pose[idx], "#f59e0b", width, height, "pc int8", legend_y)
            legend_y += 20
        if self.show_ps_float32_var.get() and self.result.ps_float32_pose is not None and idx < len(self.result.ps_float32_pose):
            self._draw_pose(self.result.ps_float32_pose[idx], "#a78bfa", width, height, "ps float32", legend_y)
            legend_y += 20
        if self.show_ps_int8_var.get() and self.result.ps_int8_pose is not None and idx < len(self.result.ps_int8_pose):
            self._draw_pose(self.result.ps_int8_pose[idx], "#ef4444", width, height, "ps int8", legend_y)
            legend_y += 20
        if self.show_fpga_pl_var.get() and idx < len(self.result.fpga_pose):
            self._draw_pose(self.result.fpga_pose[idx], "#38bdf8", width, height, "pl int8", legend_y)


def print_summary(result: RunResult) -> None:
    count = result.sample_count
    hls_values = np.asarray([item["hls_ms"] for item in result.fpga_timing if "hls_ms" in item], dtype=np.float32)
    esp_gap_values = np.asarray([item["esp_gap_ms"] for item in result.fpga_timing if item.get("esp_gap_ms", -1.0) >= 0.0], dtype=np.float32)
    esp_drop_values = np.asarray([item["esp_drop_count"] for item in result.fpga_timing if "esp_drop_count" in item], dtype=np.float32)
    esp_max_gap_values = np.asarray([item["esp_max_gap"] for item in result.fpga_timing if "esp_max_gap" in item], dtype=np.float32)
    print(f"samples={count}")
    print(f"pc_float32_windows_per_sec={count / max(result.pc_float32_elapsed_sec, 1e-9):.3f}")
    print(f"pc_int8_windows_per_sec={count / max(result.pc_int8_elapsed_sec, 1e-9):.3f}")
    print(f"fpga_windows_per_sec={len(result.fpga_pose) / max(result.fpga_elapsed_sec, 1e-9):.3f}")
    print(f"pc_float32_forward_ms_mean={float(np.mean(result.pc_float32_forward_ms)):.6f}")
    print(f"pc_float32_forward_ms_median={float(np.median(result.pc_float32_forward_ms)):.6f}")
    print(f"pc_int8_forward_ms_mean={float(np.mean(result.pc_int8_forward_ms)):.6f}")
    print(f"pc_int8_forward_ms_median={float(np.median(result.pc_int8_forward_ms)):.6f}")
    if len(hls_values):
        print(f"fpga_hls_ms_mean={float(np.mean(hls_values)):.6f}")
        print(f"fpga_hls_ms_median={float(np.median(hls_values)):.6f}")
    if len(esp_gap_values):
        print(f"esp_gap_ms_mean={float(np.mean(esp_gap_values)):.6f}")
        print(f"esp_gap_ms_median={float(np.median(esp_gap_values)):.6f}")
    if len(esp_drop_values):
        print(f"esp_drop_count_total={int(np.sum(esp_drop_values))}")
        print(f"esp_drop_count_max={int(np.max(esp_drop_values))}")
    if len(esp_max_gap_values):
        print(f"esp_max_gap_max={int(np.max(esp_max_gap_values))}")
    print(f"mae_fpga_vs_pc_float32={mae(result.fpga_pose, result.pc_float32_pose):.6f}")
    print(f"mae_fpga_vs_pc_int8={mae(result.fpga_pose, result.pc_int8_pose):.6f}")
    if result.output_npz is not None:
        print(f"output_npz={result.output_npz}")
    if result.summary_json is not None:
        print(f"summary_json={result.summary_json}")


def main() -> int:
    args = build_parser().parse_args()
    if args.no_gui:
        result = run_compare(args, status=lambda text: print(text, flush=True))
        print_summary(result)
        return 0
    app = CompareApp(args)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
