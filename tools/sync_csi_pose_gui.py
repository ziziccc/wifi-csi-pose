from __future__ import annotations

import csv
import json
import math
import queue
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import TextIO

import sync_pose_input as pose_input
from sync_csi_input import DEFAULT_BAUDRATE, ParsedRecord, SerialReader, list_serial_ports
from sync_pose_input import (
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    PoseFrameEvent,
    PoseSample,
    PoseStream,
    build_network_source_candidates,
    list_available_cameras,
)


CAPTURE_DIR = Path(__file__).resolve().parents[1] / "captures"
CSI_FIELDS = [
    "csi_host_time",
    "trigger_seq",
    "rx_index",
    "rssi",
    "csi_len",
    "iq_pairs",
]
POSE_FIELDS = [
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
    "left_elbow_x",
    "left_elbow_y",
    "right_elbow_x",
    "right_elbow_y",
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y",
    "left_hip_x",
    "left_hip_y",
    "right_hip_x",
    "right_hip_y",
    "left_knee_x",
    "left_knee_y",
    "right_knee_x",
    "right_knee_y",
    "left_ankle_x",
    "left_ankle_y",
    "right_ankle_x",
    "right_ankle_y",
]
POSE_META_FIELDS = [
    "frame_width",
    "frame_height",
]
CSV_FIELDS = CSI_FIELDS + POSE_META_FIELDS + POSE_FIELDS
DEFAULT_CSI_PORT = "COM22"
DEFAULT_NETWORK_URL = "http://172.21.101.0:8080"
DEFAULT_FLUSH_INTERVAL_SEC = 3.0
MAX_BUFFERED_CSV_ROWS = 512
MAX_CSI_EVENTS_PER_POLL = 4096
MAX_FRAME_EVENTS_PER_POLL = 64
MAX_POSE_EVENTS_PER_POLL = 64
MAX_POSE_LOGS_PER_POLL = 32
MAX_ROWS_PER_FLUSH_PASS = 4096
PREVIEW_UPDATE_INTERVAL_SEC = 0.1


@dataclass
class FrameInterval:
    start_ms: int
    end_ms: int | None
    pose_sample: PoseSample | None
    pose_resolved: bool


class CaptureSession:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.duration_sec = 0.0
        self.target_files = 1
        self.completed_files = 0
        self.deadline = 0.0
        self.flush_interval_sec = DEFAULT_FLUSH_INTERVAL_SEC
        self.base_path: Path | None = None
        self.current_output_path: Path | None = None
        self.current_handle: TextIO | None = None
        self.current_writer: csv.DictWriter | None = None
        self.write_buffer: list[dict] = []
        self.rows_written = 0
        self.last_flush_time = 0.0
        self.output_name_stem = ""
        self.pending_csi: deque[ParsedRecord] = deque()
        self.intervals: deque[FrameInterval] = deque()
        self.latest_pose_sample: PoseSample | None = None
        self.pose_capture_started = False
        self.pose_capture_start_ms: int | None = None


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("CSI + Pose Sync Capture")
        self.root.geometry("1380x960")

        self.csi_events: queue.Queue = queue.Queue()
        self.csi_reader: SerialReader | None = None
        self.pose_stream = PoseStream()
        self.capture_session = CaptureSession()
        self.rate_samples: deque[tuple[float, int]] = deque()
        self.timeout_rate_samples: deque[tuple[float, int]] = deque()
        self.preview_image = None
        self.last_preview_update = 0.0

        self.port_var = tk.StringVar(value=DEFAULT_CSI_PORT)
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUDRATE))
        self.timeout_var = tk.StringVar(value="1000")
        self.slot_gap_var = tk.StringVar(value="500")
        self.channel_var = tk.StringVar(value="6")
        self.second_var = tk.StringVar(value="above")

        self.source_type_var = tk.StringVar(value="local")
        self.local_camera_var = tk.StringVar(value="0")
        self.network_url_var = tk.StringVar(value=DEFAULT_NETWORK_URL)
        self.show_camera_var = tk.BooleanVar(value=True)

        self.duration_sec_var = tk.StringVar(value="60")
        self.file_count_var = tk.StringVar(value="1")
        self.flush_interval_var = tk.StringVar(value=str(int(DEFAULT_FLUSH_INTERVAL_SEC)))
        self.capture_path_var = tk.StringVar(value="")
        self.capture_remaining_var = tk.StringVar(value="")
        self.rate_var = tk.StringVar(value="0.0 pkt/s")
        self.timeout_rate_var = tk.StringVar(value="Timeout 0.0 pkt/s")
        self.status_var = tk.StringVar(value="Idle")
        self.tx_state_vars: dict[str, tk.StringVar] = {}
        self.tx_node_rows: dict[int, str] = {}

        self._build_ui()
        self.refresh_ports()
        self.refresh_cameras()
        self.root.after(50, self.poll)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("CaptureRemaining.TLabel", font=("Segoe UI", 16, "bold"))

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="CSI Port").pack(side="left")
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var, width=16, state="readonly")
        self.port_combo.pack(side="left", padx=4)
        ttk.Button(top, text="Refresh Ports", command=self.refresh_ports).pack(side="left", padx=4)
        ttk.Label(top, text="Baud").pack(side="left", padx=(12, 0))
        ttk.Entry(top, textvariable=self.baud_var, width=10).pack(side="left", padx=4)
        ttk.Button(top, text="Connect CSI", command=self.connect_csi).pack(side="left", padx=4)
        ttk.Button(top, text="Disconnect CSI", command=self.disconnect_csi).pack(side="left", padx=4)
        ttk.Button(top, text="Clear Log", command=self.clear_log).pack(side="left", padx=4)
        ttk.Label(top, textvariable=self.timeout_rate_var).pack(side="right")
        ttk.Label(top, textvariable=self.rate_var).pack(side="right", padx=(0, 12))
        ttk.Label(top, textvariable=self.status_var).pack(side="right", padx=(0, 12))

        csi_ctrl = ttk.LabelFrame(self.root, text="CSI Control", padding=8)
        csi_ctrl.pack(fill="x", padx=8, pady=4)
        ttk.Button(csi_ctrl, text="Status", command=lambda: self.send_csi("status")).pack(side="left", padx=4)
        ttk.Button(csi_ctrl, text="Run", command=lambda: self.send_csi("mode run")).pack(side="left", padx=4)
        ttk.Button(csi_ctrl, text="Wait", command=lambda: self.send_csi("mode wait")).pack(side="left", padx=4)
        ttk.Button(csi_ctrl, text="Save State", command=lambda: self.send_csi("save_nodes")).pack(side="left", padx=4)
        ttk.Label(csi_ctrl, text="Timeout(us)").pack(side="left", padx=(12, 2))
        ttk.Entry(csi_ctrl, textvariable=self.timeout_var, width=10).pack(side="left")
        ttk.Label(csi_ctrl, text="SlotGap(us)").pack(side="left", padx=(12, 2))
        ttk.Entry(csi_ctrl, textvariable=self.slot_gap_var, width=10).pack(side="left")
        ttk.Button(csi_ctrl, text="Apply Timing", command=self.apply_csi_timing).pack(side="left", padx=4)
        ttk.Label(csi_ctrl, text="Channel").pack(side="left", padx=(12, 2))
        ttk.Entry(csi_ctrl, textvariable=self.channel_var, width=5).pack(side="left")
        ttk.Combobox(
            csi_ctrl,
            textvariable=self.second_var,
            values=["none", "above", "below"],
            width=8,
            state="readonly",
        ).pack(side="left", padx=4)
        ttk.Button(csi_ctrl, text="Apply Channel", command=self.apply_csi_channel).pack(side="left", padx=4)

        pose_ctrl = ttk.LabelFrame(self.root, text="Camera / Pose", padding=8)
        pose_ctrl.pack(fill="x", padx=8, pady=4)
        ttk.Radiobutton(pose_ctrl, text="Local Camera", variable=self.source_type_var, value="local").pack(side="left")
        self.local_camera_combo = ttk.Combobox(pose_ctrl, textvariable=self.local_camera_var, width=16, state="readonly")
        self.local_camera_combo.pack(side="left", padx=4)
        ttk.Button(pose_ctrl, text="Refresh Cameras", command=self.refresh_cameras).pack(side="left", padx=4)
        ttk.Radiobutton(pose_ctrl, text="Network URL", variable=self.source_type_var, value="network").pack(side="left", padx=(12, 0))
        ttk.Entry(pose_ctrl, textvariable=self.network_url_var, width=36).pack(side="left", padx=4)
        ttk.Button(pose_ctrl, text="Connect Camera", command=self.connect_camera).pack(side="left", padx=4)
        ttk.Checkbutton(
            pose_ctrl,
            text="Show Camera",
            variable=self.show_camera_var,
            command=self.update_show_camera,
        ).pack(side="left", padx=(12, 0))

        capture = ttk.LabelFrame(self.root, text="Synchronized CSV Capture (CSI + Pose)", padding=8)
        capture.pack(fill="x", padx=8, pady=4)
        ttk.Label(capture, text="Seconds/File").pack(side="left")
        ttk.Entry(capture, textvariable=self.duration_sec_var, width=8).pack(side="left", padx=4)
        ttk.Label(capture, text="File Count").pack(side="left", padx=(8, 0))
        ttk.Entry(capture, textvariable=self.file_count_var, width=8).pack(side="left", padx=4)
        ttk.Label(capture, text="Flush(s)").pack(side="left", padx=(8, 0))
        ttk.Entry(capture, textvariable=self.flush_interval_var, width=6).pack(side="left", padx=4)
        ttk.Button(capture, text="Choose CSV Path", command=self.choose_capture_path).pack(side="left", padx=4)
        ttk.Button(capture, text="Start Capture", command=self.start_capture).pack(side="left", padx=4)
        ttk.Button(capture, text="Stop Capture", command=self.stop_capture).pack(side="left", padx=4)
        ttk.Label(
            capture,
            textvariable=self.capture_remaining_var,
            style="CaptureRemaining.TLabel",
        ).pack(side="left", padx=(12, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        preview_frame = ttk.LabelFrame(body, text="Pose Preview")
        body.add(preview_frame, weight=3)
        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bg="#090b0d",
            highlightthickness=1,
            highlightbackground="#2a3138",
        )
        self.preview_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        right = ttk.Panedwindow(body, orient="vertical")
        body.add(right, weight=2)

        state_frame = ttk.LabelFrame(right, text="TX State")
        right.add(state_frame, weight=2)
        self._build_tx_state_view(state_frame)

        log_frame = ttk.LabelFrame(right, text="Logs")
        right.add(log_frame, weight=3)
        self.log = ScrolledText(log_frame, height=16)
        self.log.pack(fill="both", expand=True)

        info_frame = ttk.LabelFrame(right, text="Capture Path")
        right.add(info_frame, weight=1)
        ttk.Label(info_frame, textvariable=self.capture_path_var, wraplength=420, justify="left").pack(
            fill="both", expand=True, padx=8, pady=8
        )

    def _build_tx_state_view(self, parent: ttk.Frame) -> None:
        summary_fields = [
            ("mode", "Mode"),
            ("wifi_channel", "Channel"),
            ("second_channel", "Second"),
            ("connected_count", "Connected"),
            ("active_count", "Active"),
            ("saved_count", "Saved"),
            ("timeout_us", "Timeout(us)"),
            ("udp_slot_gap_us", "SlotGap(us)"),
            ("generation", "Generation"),
            ("next_trigger_seq", "Next Trigger"),
            ("uart_seq", "UART Seq"),
            ("trigger_sent_count", "Triggers"),
            ("cycle_timeout_count", "Cycle Timeouts"),
            ("tx_mac", "TX MAC"),
        ]

        summary = ttk.Frame(parent, padding=6)
        summary.pack(fill="x")
        for index, (key, label) in enumerate(summary_fields):
            row = index // 2
            col = (index % 2) * 2
            self.tx_state_vars[key] = tk.StringVar(value="-")
            ttk.Label(summary, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=2)
            ttk.Label(summary, textvariable=self.tx_state_vars[key], width=18).grid(
                row=row,
                column=col + 1,
                sticky="w",
                padx=(0, 12),
                pady=2,
            )

        columns = ("slot", "saved", "connect", "flags", "live", "mac", "last_seen", "rx_ok", "timeout")
        self.tx_nodes = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        headings = {
            "slot": "Slot",
            "saved": "Saved",
            "connect": "Connect",
            "flags": "Flags",
            "live": "Live",
            "mac": "MAC",
            "last_seen": "Last Seen(ms)",
            "rx_ok": "RX OK",
            "timeout": "Timeout",
        }
        widths = {
            "slot": 46,
            "saved": 56,
            "connect": 64,
            "flags": 50,
            "live": 54,
            "mac": 142,
            "last_seen": 92,
            "rx_ok": 76,
            "timeout": 76,
        }
        for column in columns:
            self.tx_nodes.heading(column, text=headings[column])
            self.tx_nodes.column(column, width=widths[column], anchor="center", stretch=column == "mac")
        self.tx_nodes.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    def log_line(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log.insert("end", f"[{timestamp}] {text}\n")
        self.log.see("end")

    def clear_log(self) -> None:
        self.log.delete("1.0", "end")

    def update_tx_state(self, status: dict) -> None:
        for key, var in self.tx_state_vars.items():
            value = status.get(key, "-")
            var.set(str(value))

        seen_slots: set[int] = set()
        for node in status.get("nodes", []):
            slot = int(node.get("slot_index", 0))
            seen_slots.add(slot)
            saved_order = node.get("saved_order", 255)
            saved_text = "-" if saved_order == 255 else str(saved_order)
            live = node.get("live")
            if live is None:
                live = bool(int(node.get("flags", 0)) & 0x04)
            values = (
                slot,
                saved_text,
                node.get("connect_order", "-"),
                node.get("flags", "-"),
                "yes" if live else "no",
                node.get("mac", "-"),
                node.get("last_seen_ms", "-"),
                node.get("rx_ok", "-"),
                node.get("rx_timeout", "-"),
            )

            item_id = self.tx_node_rows.get(slot)
            if item_id and self.tx_nodes.exists(item_id):
                self.tx_nodes.item(item_id, values=values)
            else:
                self.tx_node_rows[slot] = self.tx_nodes.insert("", "end", values=values)

        for slot, item_id in list(self.tx_node_rows.items()):
            if slot not in seen_slots and self.tx_nodes.exists(item_id):
                self.tx_nodes.delete(item_id)
                del self.tx_node_rows[slot]

    def refresh_ports(self) -> None:
        ports = list_serial_ports()
        self.port_combo["values"] = ports
        if DEFAULT_CSI_PORT in ports:
            self.port_var.set(DEFAULT_CSI_PORT)
        elif ports and self.port_var.get() not in ports:
            self.port_var.set(ports[0])

    def refresh_cameras(self) -> None:
        cameras = list_available_cameras()
        labels = [f"Camera {index}" for index in cameras] or ["Camera 0"]
        self.local_camera_combo["values"] = labels
        if cameras:
            self.local_camera_var.set(str(cameras[0]))
            self.local_camera_combo.set(f"Camera {cameras[0]}")
        else:
            self.local_camera_var.set("0")
            self.local_camera_combo.set("Camera 0")

    def connect_csi(self) -> None:
        if self.csi_reader is not None:
            return
        if not self.port_var.get():
            messagebox.showwarning("Missing Port", "Select a serial port first.")
            return
        try:
            baudrate = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror("Invalid Baud", "Baudrate must be an integer.")
            return

        self.csi_reader = SerialReader(self.port_var.get(), baudrate, self.csi_events)
        self.csi_reader.start()
        self.status_var.set("CSI connected")

    def disconnect_csi(self) -> None:
        if self.csi_reader is None:
            return
        self.csi_reader.stop()
        self.csi_reader = None
        self.status_var.set("CSI disconnected")

    def send_csi(self, line: str) -> None:
        if self.csi_reader is None:
            self.log_line("CSI is not connected.")
            return
        self.csi_reader.send_line(line)
        self.log_line(f"TX > {line}")

    def apply_csi_timing(self) -> None:
        self.send_csi(f"timeout_us {self.timeout_var.get().strip()}")
        self.send_csi(f"slot_gap_us {self.slot_gap_var.get().strip()}")
        
    def apply_csi_channel(self) -> None:
        self.send_csi(f"channel {self.channel_var.get().strip()} {self.second_var.get().strip()}")

    def connect_camera(self) -> None:
        if self.source_type_var.get() == "network":
            ok, detail = self.pose_stream.connect_source("network", self.network_url_var.get().strip())
            if not ok:
                tried = build_network_source_candidates(self.network_url_var.get().strip())
                messagebox.showerror("Camera Error", detail)
                self.log_line("Camera connect failed:\n" + "\n".join(tried))
                return
            self.log_line(f"Connected network camera: {detail}")
            return

        camera_text = self.local_camera_combo.get() or self.local_camera_var.get()
        if camera_text.lower().startswith("camera "):
            camera_text = camera_text.split(" ", 1)[1]
        camera_index = int(camera_text)
        ok, detail = self.pose_stream.connect_source("local", camera_index)
        if not ok:
            messagebox.showerror("Camera Error", detail)
            return
        self.log_line(f"Connected local camera: {detail}")

    def update_show_camera(self) -> None:
        self.pose_stream.set_show_camera(self.show_camera_var.get())

    def choose_capture_path(self) -> None:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Choose base CSV path",
            initialdir=str(CAPTURE_DIR),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if path:
            self.capture_path_var.set(path)

    def start_capture(self) -> None:
        try:
            duration_seconds = float(self.duration_sec_var.get())
            file_count = int(self.file_count_var.get())
            flush_interval_seconds = float(self.flush_interval_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Seconds, file count, and flush interval must be numeric.")
            return

        if duration_seconds <= 0.0 or file_count <= 0 or flush_interval_seconds <= 0.0:
            messagebox.showerror("Invalid Input", "Seconds, file count, and flush interval must be greater than 0.")
            return

        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        if not self.capture_path_var.get().strip():
            default_path = CAPTURE_DIR / f"sync_csi_pose_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            self.capture_path_var.set(str(default_path))

        if (
            self.capture_session.active
            or self.capture_session.current_handle is not None
            or self.capture_session.pending_csi
            or self.capture_session.write_buffer
        ):
            self._finalize_current_file(final_stop=True)

        self.capture_session.reset()
        self.capture_session.active = True
        self.capture_session.duration_sec = duration_seconds
        self.capture_session.target_files = file_count
        self.capture_session.deadline = time.time() + self.capture_session.duration_sec
        self.capture_session.flush_interval_sec = flush_interval_seconds
        self.capture_session.base_path = Path(self.capture_path_var.get().strip())
        self.capture_session.output_name_stem = self._build_unique_output_name_stem(
            self.capture_session.base_path,
            self.capture_session.target_files,
        )
        self._open_output_file_for_current_capture()
        self.log_line(
            f"Capture started: {duration_seconds:.2f} sec/file x {file_count} file(s), flush {flush_interval_seconds:.2f}s -> {self.capture_session.current_output_path}"
        )

    def stop_capture(self) -> None:
        session = self.capture_session
        if (
            not session.active
            and not session.write_buffer
            and not session.pending_csi
            and session.current_handle is None
        ):
            return
        self._finalize_current_file(final_stop=True)

    def _build_output_path(self, file_index: int) -> Path:
        assert self.capture_session.base_path is not None
        suffix = self.capture_session.base_path.suffix or ".csv"
        stem = self.capture_session.output_name_stem or self.capture_session.base_path.stem
        if self.capture_session.target_files <= 1:
            return self.capture_session.base_path.with_name(f"{stem}{suffix}")
        return self.capture_session.base_path.with_name(f"{stem}_{file_index:03d}{suffix}")

    def _build_unique_output_name_stem(self, base_path: Path, target_files: int) -> str:
        suffix = base_path.suffix or ".csv"
        timestamped_stem = f"{base_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        candidate_stem = timestamped_stem
        duplicate_index = 1

        while True:
            if target_files <= 1:
                candidate_path = base_path.with_name(f"{candidate_stem}{suffix}")
            else:
                candidate_path = base_path.with_name(f"{candidate_stem}_001{suffix}")
            if not candidate_path.exists():
                return candidate_stem
            candidate_stem = f"{timestamped_stem}_{duplicate_index:02d}"
            duplicate_index += 1

    def _pose_row(self, pose_sample: PoseSample | None) -> dict:
        row = {field: "" for field in POSE_META_FIELDS + POSE_FIELDS}
        if pose_sample is None:
            return row
        row["frame_width"] = pose_sample.frame_width
        row["frame_height"] = pose_sample.frame_height
        for key, value in pose_sample.landmarks.items():
            x, y = value
            row[f"{key}_x"] = "" if x is None else x
            row[f"{key}_y"] = "" if y is None else y
        return row

    def _csi_row(self, record: ParsedRecord, pose_sample: PoseSample | None) -> dict:
        row = {
            "csi_host_time": datetime.fromtimestamp(record.host_time).isoformat(timespec="milliseconds"),
            "trigger_seq": record.trigger_seq,
            "rx_index": record.rx_index,
            "rssi": record.rssi,
            "csi_len": record.csi_len,
            "iq_pairs": json.dumps(record.iq_pairs, ensure_ascii=False),
        }
        row.update(self._pose_row(pose_sample))
        return row

    def _open_output_file_for_current_capture(self) -> None:
        session = self.capture_session
        output_path = self._build_output_path(session.completed_files + 1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
        session.current_output_path = output_path
        session.current_handle = handle
        session.current_writer = writer
        self._reset_capture_state_for_new_file()
        session.last_flush_time = time.monotonic()

    def _reset_capture_state_for_new_file(self) -> None:
        session = self.capture_session
        session.pending_csi.clear()
        session.intervals.clear()
        session.write_buffer = []
        session.rows_written = 0
        session.last_flush_time = 0.0
        session.latest_pose_sample = None
        session.pose_capture_started = False
        session.pose_capture_start_ms = None

    def _flush_csv_buffer(self, force_disk_flush: bool) -> None:
        session = self.capture_session
        if session.current_handle is None or session.current_writer is None:
            return

        if session.write_buffer:
            session.current_writer.writerows(session.write_buffer)
            session.rows_written += len(session.write_buffer)
            session.write_buffer.clear()

        now = time.monotonic()
        if force_disk_flush or (now - session.last_flush_time) >= session.flush_interval_sec:
            session.current_handle.flush()
            session.last_flush_time = now

    def _queue_row_for_write(self, row: dict) -> None:
        session = self.capture_session
        session.write_buffer.append(row)
        if len(session.write_buffer) >= MAX_BUFFERED_CSV_ROWS:
            self._flush_csv_buffer(force_disk_flush=False)

    def _prune_intervals(self) -> None:
        session = self.capture_session
        while len(session.intervals) > 1:
            next_interval_start = session.intervals[1].start_ms
            if session.pending_csi and session.pending_csi[0].monotonic_ms < next_interval_start:
                break
            session.intervals.popleft()

    def _flush_resolved_rows(self, final_flush: bool, max_rows: int | None = None) -> None:
        session = self.capture_session
        rows_processed = 0

        while session.pending_csi:
            if max_rows is not None and rows_processed >= max_rows:
                break
            record = session.pending_csi[0]

            # 1) 카메라 미사용 등으로 Pose 구간 정보가 없으면 CSI만 즉시 기록
            if not session.intervals:
                self._queue_row_for_write(self._csi_row(record, session.latest_pose_sample))
                session.pending_csi.popleft()
                rows_processed += 1
                continue

            # 2) 수집된 구간들보다 과거 시점의 CSI 데이터일 경우 즉시 기록
            if record.monotonic_ms < session.intervals[0].start_ms:
                self._queue_row_for_write(self._csi_row(record, session.latest_pose_sample))
                session.pending_csi.popleft()
                rows_processed += 1
                continue

            matching_interval = None
            for interval in session.intervals:
                if record.monotonic_ms < interval.start_ms:
                    break
                interval_end = float("inf") if interval.end_ms is None else interval.end_ms
                if interval.start_ms <= record.monotonic_ms < interval_end:
                    matching_interval = interval
                    break

            if matching_interval is None:
                if final_flush:
                    self._queue_row_for_write(self._csi_row(record, session.latest_pose_sample))
                    session.pending_csi.popleft()
                    rows_processed += 1
                    continue
                break

            if not final_flush and matching_interval.end_ms is None:
                break
            if not final_flush and not matching_interval.pose_resolved:
                break

            pose_sample = matching_interval.pose_sample
            if pose_sample is None:
                pose_sample = session.latest_pose_sample
                
            self._queue_row_for_write(self._csi_row(record, pose_sample))
            session.pending_csi.popleft()
            rows_processed += 1

        self._prune_intervals()

    def _finalize_current_file(self, final_stop: bool) -> None:
        session = self.capture_session
        if session.latest_pose_sample is not None and session.intervals:
            last_interval = session.intervals[-1]
            if last_interval.pose_sample is None:
                last_interval.pose_sample = session.latest_pose_sample
                last_interval.pose_resolved = True

        self._flush_resolved_rows(final_flush=True)
        self._flush_csv_buffer(force_disk_flush=True)

        saved_path = session.current_output_path
        saved_rows = session.rows_written
        if session.current_handle is not None:
            session.current_handle.close()
            session.current_handle = None
            session.current_writer = None
            session.completed_files += 1
            self.log_line(
                f"Saved [{session.completed_files}/{session.target_files}]: {saved_path} ({saved_rows} rows)"
            )

        session.current_output_path = None
        self._reset_capture_state_for_new_file()

        if final_stop or session.completed_files >= session.target_files:
            self.capture_remaining_var.set("")
            self.log_line("Capture finished")
            session.reset()
            return

        session.deadline = time.time() + session.duration_sec
        self._open_output_file_for_current_capture()
        self.log_line(f"Starting next file [{session.completed_files + 1}/{session.target_files}]")

    def _append_pose_frame_event(self, frame_event: PoseFrameEvent) -> None:
        session = self.capture_session
        if not session.active:
            return
        if session.intervals:
            session.intervals[-1].end_ms = frame_event.frame_timestamp_ms
        session.intervals.append(
            FrameInterval(
                start_ms=frame_event.frame_timestamp_ms,
                end_ms=None,
                pose_sample=None,
                pose_resolved=False,
            )
        )

    def _append_pose_sample(self, pose_sample: PoseSample) -> None:
        session = self.capture_session
        session.latest_pose_sample = pose_sample
        if not session.active:
            return
        for interval in reversed(session.intervals):
            if interval.start_ms == pose_sample.frame_timestamp_ms:
                interval.pose_sample = pose_sample
                interval.pose_resolved = True
                if not session.pose_capture_started:
                    session.pose_capture_started = True
                    session.pose_capture_start_ms = pose_sample.frame_timestamp_ms
                    self.log_line("Pose acquired; synchronizing with CSI data.")
                break
            if interval.start_ms < pose_sample.frame_timestamp_ms:
                break

    def _append_csi_record(self, record: ParsedRecord) -> None:
        now = time.monotonic()
        self.rate_samples.append((now, record.record_bytes))
        if self.capture_session.active:
            self.capture_session.pending_csi.append(record)

    def _update_rates(self) -> None:
        now = time.monotonic()
        cutoff = now - 1.0
        while self.rate_samples and self.rate_samples[0][0] < cutoff:
            self.rate_samples.popleft()
        while self.timeout_rate_samples and self.timeout_rate_samples[0][0] < cutoff:
            self.timeout_rate_samples.popleft()

        self.rate_var.set(f"{float(len(self.rate_samples)):.1f} pkt/s")
        timeout_rate = float(sum(timeout_count for _, timeout_count in self.timeout_rate_samples))
        self.timeout_rate_var.set(f"Timeout {timeout_rate:.1f} pkt/s")

    def _update_remaining(self) -> None:
        session = self.capture_session
        if not session.active:
            self.capture_remaining_var.set("")
            return

        remaining = max(0.0, session.deadline - time.time())
        self.capture_remaining_var.set(
            f"Remaining: {remaining:.1f}s / File {session.completed_files + 1}/{session.target_files}"
        )
        if time.time() >= session.deadline:
            self._finalize_current_file(final_stop=False)

    def _update_preview(self) -> None:
        frame = self.pose_stream.get_preview_frame()
        self.preview_canvas.delete("all")
        canvas_width = max(self.preview_canvas.winfo_width(), DISPLAY_WIDTH)
        canvas_height = max(self.preview_canvas.winfo_height(), DISPLAY_HEIGHT)
        if frame is None:
            self.preview_canvas.create_text(
                canvas_width / 2,
                canvas_height / 2,
                text="No camera frame",
                fill="#cfd8dc",
                font=("Segoe UI", 16, "bold"),
            )
            self.preview_image = None
            return

        frame_height, frame_width = frame.shape[:2]
        scale = max(canvas_width / max(frame_width, 1), canvas_height / max(frame_height, 1))
        scaled_width = max(1, int(math.ceil(frame_width * scale)))
        scaled_height = max(1, int(math.ceil(frame_height * scale)))
        scaled = pose_input.cv2.resize(
            frame,
            (scaled_width, scaled_height),
            interpolation=pose_input.cv2.INTER_LINEAR,
        )
        x_offset = max(0, (scaled_width - canvas_width) // 2)
        y_offset = max(0, (scaled_height - canvas_height) // 2)
        resized = scaled[y_offset : y_offset + canvas_height, x_offset : x_offset + canvas_width]
        rgb = pose_input.cv2.cvtColor(resized, pose_input.cv2.COLOR_BGR2RGB)
        ppm_header = f"P6\n{canvas_width} {canvas_height}\n255\n".encode("ascii")
        ppm_data = ppm_header + rgb.tobytes()
        self.preview_image = tk.PhotoImage(data=ppm_data, format="PPM")
        self.preview_canvas.create_image(0, 0, image=self.preview_image, anchor="nw")

    def poll(self) -> None:
        processed_csi_events = 0
        try:
            while processed_csi_events < MAX_CSI_EVENTS_PER_POLL:
                kind, payload = self.csi_events.get_nowait()
                if kind == "log":
                    self.log_line(str(payload))
                elif kind == "closed":
                    self.status_var.set("CSI disconnected")
                    self.csi_reader = None
                elif kind == "record":
                    self._append_csi_record(payload)
                elif kind == "cycle":
                    if payload.timeout_packets:
                        self.timeout_rate_samples.append((time.monotonic(), payload.timeout_packets))
                elif kind == "status":
                    self.update_tx_state(payload)
                elif kind == "ack":
                    self.log_line(json.dumps(payload, ensure_ascii=False))
                processed_csi_events += 1
        except queue.Empty:
            pass

        frame_events, pose_samples, pose_logs = self.pose_stream.drain_events(
            max_frame_events=MAX_FRAME_EVENTS_PER_POLL,
            max_pose_events=MAX_POSE_EVENTS_PER_POLL,
            max_log_events=MAX_POSE_LOGS_PER_POLL,
        )
        for log_line in pose_logs:
            self.log_line(log_line)
        for frame_event in frame_events:
            self._append_pose_frame_event(frame_event)
        for pose_sample in pose_samples:
            self._append_pose_sample(pose_sample)

        self._flush_resolved_rows(final_flush=False, max_rows=MAX_ROWS_PER_FLUSH_PASS)
        self._flush_csv_buffer(force_disk_flush=False)
        self._update_rates()
        self._update_remaining()
        now = time.monotonic()
        if (now - self.last_preview_update) >= PREVIEW_UPDATE_INTERVAL_SEC:
            self._update_preview()
            self.last_preview_update = now
        self.root.after(50, self.poll)

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            session = self.capture_session
            if session.active or session.current_handle is not None or session.pending_csi or session.write_buffer:
                self._finalize_current_file(final_stop=True)
            if self.csi_reader is not None:
                self.csi_reader.stop()
            self.pose_stream.close()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()