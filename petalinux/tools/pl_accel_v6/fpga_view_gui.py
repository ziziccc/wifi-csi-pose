from __future__ import annotations

import argparse
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import messagebox, ttk


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


POSE_PIXEL_WIDTH = 640
POSE_PIXEL_HEIGHT = 480
POSE_AREA_OUTLINE = "#2e3440"


@dataclass
class PoseSample:
    values: list[float]
    index: int
    raw: str
    received_at: str
    receive_gap_ms: float | None
    gap_source: str
    timing_text: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live FPGA pose viewer over SSH.")
    parser.add_argument("--host", default="192.168.1.15")
    parser.add_argument("--user", default="root")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--tty", default="/dev/ttyACM0")
    parser.add_argument("--remote-dir", default="/home/root")
    parser.add_argument("--runner", default="./esp_pose_pl_runner_rt")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--stdin", action="store_true", help="Read pose CSV lines from local stdin instead of SSH.")
    parser.add_argument(
        "--remote-cmd",
        default=None,
        help="Override remote command. It must print pose lines containing 24 comma-separated floats.",
    )
    return parser


def parse_pose_line(line: str) -> list[float] | None:
    text = line.strip()
    if not text:
        return None
    if "[" in text and "]" in text:
        text = text[text.find("[") + 1 : text.rfind("]")]
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 24:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def parse_timing_metrics(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=(-?\d+(?:\.\d+)?)", line):
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def format_timing_text(metrics: dict[str, float]) -> str:
    fields = [
        ("esp_gap", "esp_gap_ms"),
        ("esp_drop", "esp_drop_count"),
        ("esp_max_gap", "esp_max_gap"),
        ("usb", "usb_ms"),
        ("data_ready", "data_ready_ms"),
        ("data", "data_ms"),
        ("queue", "queue_ms"),
        ("pl_in", "pl_in_ms"),
        ("pl_out", "pl_out_ms"),
        ("hls", "hls_ms"),
        ("total", "total_ms"),
    ]
    parts: list[str] = []
    for label, key in fields:
        if key not in metrics:
            continue
        if key.endswith("_ms"):
            parts.append(f"{label}={metrics[key]:.1f}ms")
        else:
            parts.append(f"{label}={int(metrics[key])}")
    if metrics.get("dropped_old", 0.0) >= 1.0:
        parts.append("dropped_old=1")
    return "  ".join(parts)


class PoseReader(threading.Thread):
    def __init__(self, args: argparse.Namespace, out_queue: queue.Queue[PoseSample | str]) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.out_queue = out_queue
        self.stop_requested = threading.Event()
        self.process: subprocess.Popen[str] | None = None

    def run(self) -> None:
        try:
            if self.args.stdin:
                self._read_stream(sys.stdin)
            else:
                self._read_ssh()
        except Exception as exc:  # noqa: BLE001
            self.out_queue.put(f"reader error: {exc}")

    def stop(self) -> None:
        self.stop_requested.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if not self.args.stdin:
            self._kill_remote_runner()

    def _kill_remote_runner(self) -> None:
        target = f"{self.args.user}@{self.args.host}"
        command = [
            "ssh",
            "-p",
            str(self.args.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            target,
            (
                "for p in $(ps w | awk '/[e]sp_pose_pl_runner_rt/ {print $1}'); "
                "do kill $p 2>/dev/null; done; "
                "sleep 0.2; "
                "for p in $(ps w | awk '/[e]sp_pose_pl_runner_rt/ {print $1}'); "
                "do kill -9 $p 2>/dev/null; done"
            ),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        except Exception:
            pass

    def _remote_command(self) -> str:
        if self.args.remote_cmd:
            return self.args.remote_cmd
        remote_dir = shlex.quote(self.args.remote_dir)
        runner = shlex.quote(self.args.runner)
        tty = shlex.quote(self.args.tty)
        print_every = shlex.quote(str(self.args.print_every))
        return (
            f"cd {remote_dir} || exit 120; "
            f"if [ ! -e {runner} ]; then echo 'runner not found: {self.args.runner}'; exit 121; fi; "
            f"if [ ! -s {runner} ]; then echo 'runner is empty: {self.args.runner}'; exit 122; fi; "
            f"if [ ! -x {runner} ]; then echo 'runner is not executable: {self.args.runner}'; exit 123; fi; "
            f"if [ ! -e {tty} ]; then echo 'tty not found: {self.args.tty}'; ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null; exit 124; fi; "
            "for p in $(ps w | awk '/[e]sp_pose_pl_runner_rt/ {print $1}'); do kill $p 2>/dev/null; done; "
            "sleep 0.2; "
            "for p in $(ps w | awk '/[e]sp_pose_pl_runner_rt/ {print $1}'); do kill -9 $p 2>/dev/null; done; "
            f"exec {runner} --tty {tty} --no-esp-config --print-every {print_every}"
        )

    def _read_ssh(self) -> None:
        target = f"{self.args.user}@{self.args.host}"
        command = [
            "ssh",
            "-p",
            str(self.args.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            target,
            self._remote_command(),
        ]
        self.out_queue.put("running: " + " ".join(command))
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if self.process.stdout is None:
            self.out_queue.put("ssh stdout unavailable")
            return
        self._read_stream(self.process.stdout)
        rc = self.process.wait()
        self.out_queue.put(f"ssh exited: {rc}")

    def _read_stream(self, stream) -> None:
        sample_index = 0
        previous_received = None
        for line in stream:
            if self.stop_requested.is_set():
                break
            values = parse_pose_line(line)
            if values is None:
                text = line.strip()
                if text:
                    self.out_queue.put(text)
                continue
            sample_index += 1
            metrics = parse_timing_metrics(line)
            received = time.monotonic()
            host_gap_ms = None
            if previous_received is not None:
                host_gap_ms = (received - previous_received) * 1000.0
            previous_received = received
            esp_gap_ms = metrics.get("esp_gap_ms")
            runner_gap_ms = metrics.get("gap_ms")
            if esp_gap_ms is not None and esp_gap_ms >= 0.0:
                receive_gap_ms = esp_gap_ms
                gap_source = "esp"
            elif runner_gap_ms is not None and runner_gap_ms >= 0.0:
                receive_gap_ms = runner_gap_ms
                gap_source = "pl_out"
            else:
                receive_gap_ms = host_gap_ms
                gap_source = "host"
            received_at = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.out_queue.put(
                PoseSample(
                    values=values,
                    index=sample_index,
                    raw=line.strip(),
                    received_at=received_at,
                    receive_gap_ms=receive_gap_ms,
                    gap_source=gap_source,
                    timing_text=format_timing_text(metrics),
                )
            )


class SshCheck(threading.Thread):
    def __init__(self, args: argparse.Namespace, out_queue: queue.Queue[PoseSample | str]) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.out_queue = out_queue

    def run(self) -> None:
        if self.args.stdin:
            self.out_queue.put("stdin mode: ready")
            return
        target = f"{self.args.user}@{self.args.host}"
        command = [
            "ssh",
            "-p",
            str(self.args.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "-o",
            "StrictHostKeyChecking=accept-new",
            target,
            "echo connected",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=7, check=False)
        except Exception as exc:  # noqa: BLE001
            self.out_queue.put(f"ssh disconnected: {exc}")
            return
        if result.returncode == 0 and "connected" in result.stdout:
            self.out_queue.put(f"ssh connected: {target}:{self.args.port}")
        else:
            message = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
            self.out_queue.put(f"ssh disconnected: {message}")


class LivePoseViewer(tk.Tk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.queue: queue.Queue[PoseSample | str] = queue.Queue()
        self.reader: PoseReader | None = None
        self.running = False
        self.last_pose: list[float] | None = None
        self.sample_count = 0
        self.received_at = ""
        self.receive_gap_ms: float | None = None
        self.gap_source = "host"
        self.timing_text = ""

        self.title("FPGA Live Pose Viewer")
        self.geometry("980x760")
        self.minsize(720, 560)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="starting")
        self.receive_var = tk.StringVar(value="")
        self.raw_var = tk.StringVar(value="")

        top = ttk.Frame(self, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.start_button = ttk.Button(top, text="Start", command=self._start_reader)
        self.start_button.grid(row=0, column=1, padx=(8, 0))
        self.stop_button = ttk.Button(top, text="Stop", command=self._stop_reader, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=(8, 0))

        self.canvas = tk.Canvas(self, background="#111318", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw())

        bottom = ttk.Frame(self, padding=(10, 8))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.receive_var).grid(row=0, column=0, sticky="ew")
        ttk.Label(bottom, textvariable=self.raw_var).grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        SshCheck(args, self.queue).start()
        self.after(30, self._poll_queue)

    def _start_reader(self) -> None:
        if self.running:
            return
        self.last_pose = None
        self.sample_count = 0
        self.received_at = ""
        self.receive_gap_ms = None
        self.gap_source = "host"
        self.timing_text = ""
        self.receive_var.set("")
        self.raw_var.set("")
        self.reader = PoseReader(self.args, self.queue)
        self.reader.start()
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("running")
        self._draw()

    def _stop_reader(self) -> None:
        if self.reader is not None:
            self.reader.stop()
        self.running = False
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("stopped")

    def _on_close(self) -> None:
        if self.reader is not None:
            self.reader.stop()
        self.destroy()

    def _poll_queue(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, PoseSample):
                self.last_pose = item.values
                self.sample_count = item.index
                self.received_at = item.received_at
                self.receive_gap_ms = item.receive_gap_ms
                self.gap_source = item.gap_source
                self.timing_text = item.timing_text
                receive_gap_text = "first sample" if item.receive_gap_ms is None else f"{item.receive_gap_ms:.1f} ms"
                self.status_var.set(
                    f"running: samples={item.index} host={self.args.host} gap({item.gap_source})={receive_gap_text}"
                )
                timing_suffix = f"  {item.timing_text}" if item.timing_text else ""
                self.receive_var.set(
                    f"received_at={item.received_at}  gap({item.gap_source})={receive_gap_text}{timing_suffix}"
                )
                self.raw_var.set(item.raw)
                self._draw()
            else:
                self.status_var.set(item)
                if item.startswith(("runner ", "tty not found")):
                    self.raw_var.set(item)
                if item.startswith("ssh exited"):
                    self.running = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
        self.after(30, self._poll_queue)

    def _pose_origin(self, width: int, height: int) -> tuple[float, float]:
        """Return the top-left point of the fixed 640x480 pose area."""
        origin_x = max((width - POSE_PIXEL_WIDTH) / 2.0, 0.0)
        origin_y = max((height - POSE_PIXEL_HEIGHT) / 2.0, 0.0)
        return origin_x, origin_y

    def _pose_points(self, pose: list[float], origin_x: float, origin_y: float) -> dict[str, tuple[float, float]]:
        points: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(JOINT_NAMES):
            pose_x = pose[i * 2]
            pose_y = pose[i * 2 + 1]

            # The runner sends normalized coordinates in the range 0.0 ~ 1.0.
            # Convert them to fixed 640x480 pixel coordinates.
            # This no longer depends on the current GUI canvas size.
            x = origin_x + pose_x * POSE_PIXEL_WIDTH
            y = origin_y + pose_y * POSE_PIXEL_HEIGHT
            points[name] = (x, y)
        return points

    def _draw(self) -> None:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        origin_x, origin_y = self._pose_origin(width, height)
        frame_x2 = origin_x + POSE_PIXEL_WIDTH
        frame_y2 = origin_y + POSE_PIXEL_HEIGHT

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, height, fill="#111318", outline="")
        self.canvas.create_rectangle(origin_x, origin_y, frame_x2, frame_y2, outline=POSE_AREA_OUTLINE, width=2)
        self.canvas.create_text(
            origin_x + 8,
            origin_y + 8,
            text=f"fixed pose area: {POSE_PIXEL_WIDTH}x{POSE_PIXEL_HEIGHT}px",
            fill="#8f98a8",
            anchor="nw",
            font=("Segoe UI", 10),
        )

        if self.last_pose is None:
            self.canvas.create_text(
                origin_x + POSE_PIXEL_WIDTH / 2,
                origin_y + POSE_PIXEL_HEIGHT / 2,
                text="waiting for pose CSV lines",
                fill="#d8dee9",
                font=("Segoe UI", 16),
            )
            return

        points = self._pose_points(self.last_pose, origin_x, origin_y)
        for a, b in SKELETON_EDGES:
            ax, ay = points[a]
            bx, by = points[b]
            self.canvas.create_line(ax, ay, bx, by, fill="#00d0ff", width=4, capstyle=tk.ROUND)
        for name, (x, y) in points.items():
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#ffd166", outline="#111318", width=2)

        self.canvas.create_text(
            14,
            14,
            text=f"sample {self.sample_count}",
            fill="#d8dee9",
            anchor="nw",
            font=("Segoe UI", 12, "bold"),
        )
        receive_gap_text = "first sample" if self.receive_gap_ms is None else f"{self.receive_gap_ms:.1f} ms"
        self.canvas.create_text(
            14,
            38,
            text=f"received {self.received_at}  gap({self.gap_source}) {receive_gap_text}",
            fill="#d8dee9",
            anchor="nw",
            font=("Segoe UI", 11),
        )
        if self.timing_text:
            self.canvas.create_text(
                14,
                62,
                text=self.timing_text,
                fill="#d8dee9",
                anchor="nw",
                font=("Segoe UI", 10),
            )


def main() -> int:
    args = build_parser().parse_args()
    try:
        app = LivePoseViewer(args)
        app.mainloop()
    except FileNotFoundError as exc:
        messagebox.showerror("FPGA Live Pose Viewer", f"Command not found: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
