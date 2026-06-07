from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import urllib.request
from urllib.parse import urlparse, urlunparse


cv2 = None
mp = None
vision = None

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = SCRIPT_DIR / "models" / "pose_landmarker_lite.task"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
TARGET_FPS = 30
LOCAL_CAMERA_SCAN_LIMIT = 8
COMMON_NETWORK_PATHS = ["/video", "/video_feed", "/mjpeg", "/stream", "/live"]
NETWORK_DRAIN_GRABS = 2
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 200, 0)
POSE_KEYPOINTS = [
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_hip", 23),
    ("right_hip", 24),
    ("left_knee", 25),
    ("right_knee", 26),
    ("left_ankle", 27),
    ("right_ankle", 28),
]
VISIBLE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def load_runtime_dependencies() -> None:
    global cv2, mp, vision

    if cv2 is not None and mp is not None and vision is not None:
        return

    try:
        import cv2 as _cv2
        import mediapipe as _mp
        from mediapipe.tasks.python import vision as _vision
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: python -m pip install mediapipe opencv-python"
        ) from exc

    cv2 = _cv2
    mp = _mp
    vision = _vision


def ensure_model_exists() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def try_open_capture(source):
    if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def configure_capture(cap, source_type: str) -> None:
    for prop_name, value in (
        ("CAP_PROP_FRAME_WIDTH", DISPLAY_WIDTH),
        ("CAP_PROP_FRAME_HEIGHT", DISPLAY_HEIGHT),
        ("CAP_PROP_FPS", TARGET_FPS),
        ("CAP_PROP_BUFFERSIZE", 1),
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", 3000),
        ("CAP_PROP_READ_TIMEOUT_MSEC", 1000),
    ):
        prop_id = getattr(cv2, prop_name, None)
        if prop_id is None:
            continue
        try:
            cap.set(prop_id, value)
        except Exception:
            pass

    if source_type == "local":
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass


def list_available_cameras(max_index: int = LOCAL_CAMERA_SCAN_LIMIT) -> list[int]:
    load_runtime_dependencies()
    available = []
    for camera_index in range(max_index):
        cap = try_open_capture(camera_index)
        if cap is None:
            continue
        available.append(camera_index)
        cap.release()
    return available


def build_network_source_candidates(raw_url: str) -> list[str]:
    raw_url = raw_url.strip()
    if not raw_url:
        return []

    parsed = urlparse(raw_url)
    if not parsed.scheme:
        parsed = urlparse(f"http://{raw_url}")

    normalized = urlunparse(parsed)
    candidates = [normalized]
    if parsed.path in ("", "/"):
        for path in COMMON_NETWORK_PATHS:
            candidate = urlunparse(parsed._replace(path=path, params="", query="", fragment=""))
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def read_latest_available_frame(cap, source_type: str):
    if source_type != "network":
        return cap.read()

    grabbed = False
    for _ in range(NETWORK_DRAIN_GRABS):
        try:
            if not cap.grab():
                break
            grabbed = True
        except Exception:
            break

    if grabbed:
        return cap.retrieve()
    return cap.read()


def draw_pose_overlay(frame, landmarks: dict[str, tuple[int | None, int | None]] | None) -> None:
    if not landmarks:
        return

    point_map = {name: landmarks.get(name) for name, _ in POSE_KEYPOINTS}
    index_name_map = {index: name for name, index in POSE_KEYPOINTS}

    for start_index, end_index in VISIBLE_CONNECTIONS:
        start_name = index_name_map[start_index]
        end_name = index_name_map[end_index]
        start = point_map.get(start_name)
        end = point_map.get(end_name)
        if start is None or end is None:
            continue
        if None in start or None in end:
            continue
        cv2.line(frame, start, end, CONNECTION_COLOR, 2)

    for name, _ in POSE_KEYPOINTS:
        point = point_map.get(name)
        if point is None or None in point:
            continue
        cv2.circle(frame, point, 4, LANDMARK_COLOR, -1)


@dataclass
class PoseFrameEvent:
    frame_timestamp_ms: int
    frame_width: int
    frame_height: int


@dataclass
class PoseSample:
    frame_timestamp_ms: int
    captured_at: str
    frame_width: int
    frame_height: int
    landmarks: dict[str, tuple[int | None, int | None]]


@dataclass
class FramePacket:
    captured_monotonic_ms: int
    frame: object


class PoseStream:
    def __init__(self) -> None:
        load_runtime_dependencies()
        model_path = ensure_model_exists()
        self.frame_events: queue.Queue = queue.Queue()
        self.pose_events: queue.Queue = queue.Queue()
        self.log_events: queue.Queue = queue.Queue()
        self.capture = None
        self.source_type = "local"
        self.source_value: int | str = 0
        self.show_camera = True
        self.stop_event = threading.Event()
        self.capture_lock = threading.Lock()
        self.preview_lock = threading.Lock()
        self.latest_frame = None
        self.latest_pose: PoseSample | None = None
        self.latest_packet: FramePacket | None = None
        self.pending_frame_sizes: dict[int, tuple[int, int]] = {}
        self.last_consumed_capture_ms = -1
        self.inference_busy = False
        self.last_submitted_timestamp_ms = -1
        self.options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_pose_result,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(self.options)
        self.capture_worker = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_worker = threading.Thread(target=self._inference_loop, daemon=True)
        self.capture_worker.start()
        self.inference_worker.start()

    def close(self) -> None:
        self.stop_event.set()
        with self.capture_lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
        self.capture_worker.join(timeout=2.0)
        self.inference_worker.join(timeout=2.0)
        self.landmarker.close()

    def connect_source(self, source_type: str, source_value: int | str) -> tuple[bool, str]:
        candidates = [source_value]
        resolved_source: int | str = source_value
        if source_type == "network":
            candidates = build_network_source_candidates(str(source_value))

        new_capture = None
        tried = []
        for candidate in candidates:
            tried.append(str(candidate))
            candidate_capture = try_open_capture(candidate)
            if candidate_capture is None:
                continue
            configure_capture(candidate_capture, source_type)
            new_capture = candidate_capture
            resolved_source = candidate
            break

        if new_capture is None:
            return False, "Could not open source. Tried:\n" + "\n".join(tried)

        with self.capture_lock:
            if self.capture is not None:
                self.capture.release()
            self.capture = new_capture
            self.source_type = source_type
            self.source_value = resolved_source

        with self.preview_lock:
            self.latest_frame = None
            self.latest_pose = None
            self.latest_packet = None
            self.pending_frame_sizes.clear()
            self.last_consumed_capture_ms = -1
            self.last_submitted_timestamp_ms = -1
            self.inference_busy = False
        return True, str(resolved_source)

    def set_show_camera(self, visible: bool) -> None:
        self.show_camera = visible

    def get_preview_frame(self):
        with self.preview_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            pose = self.latest_pose

        if frame is None:
            return None
        if not self.show_camera:
            frame[:] = 0
        draw_pose_overlay(frame, None if pose is None else pose.landmarks)
        return frame

    def drain_events(
        self,
        max_frame_events: int | None = None,
        max_pose_events: int | None = None,
        max_log_events: int | None = None,
    ) -> tuple[list[PoseFrameEvent], list[PoseSample], list[str]]:
        frame_events = []
        pose_events = []
        log_events = []

        while max_frame_events is None or len(frame_events) < max_frame_events:
            try:
                frame_events.append(self.frame_events.get_nowait())
            except queue.Empty:
                break

        while max_pose_events is None or len(pose_events) < max_pose_events:
            try:
                pose_events.append(self.pose_events.get_nowait())
            except queue.Empty:
                break

        while max_log_events is None or len(log_events) < max_log_events:
            try:
                log_events.append(self.log_events.get_nowait())
            except queue.Empty:
                break

        return frame_events, pose_events, log_events

    def _on_pose_result(self, result, _output_image, timestamp_ms: int) -> None:
        with self.preview_lock:
            frame_size = self.pending_frame_sizes.pop(timestamp_ms, None)
            frame = None
            if frame_size is None and self.latest_frame is not None:
                frame = self.latest_frame.copy()
            self.inference_busy = False

        if frame_size is not None:
            frame_width, frame_height = frame_size
        else:
            if frame is None:
                return
            frame_height, frame_width = frame.shape[:2]

        landmarks = {}
        if result.pose_landmarks:
            pose_landmarks = result.pose_landmarks[0]
            for name, landmark_index in POSE_KEYPOINTS:
                landmark = pose_landmarks[landmark_index]
                landmarks[name] = (
                    int(landmark.x * frame_width),
                    int(landmark.y * frame_height),
                )
        else:
            for name, _ in POSE_KEYPOINTS:
                landmarks[name] = (None, None)

        sample = PoseSample(
            frame_timestamp_ms=timestamp_ms,
            captured_at=datetime_now_iso(),
            frame_width=frame_width,
            frame_height=frame_height,
            landmarks=landmarks,
        )
        with self.preview_lock:
            self.latest_pose = sample
        self.pose_events.put(sample)

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.capture_lock:
                capture = self.capture
                source_type = self.source_type

            if capture is None:
                time.sleep(0.03)
                continue

            success, frame = read_latest_available_frame(capture, source_type)
            if not success:
                self.log_events.put("Camera frame read failed")
                time.sleep(0.05)
                continue

            if source_type == "local":
                frame = cv2.flip(frame, 1)

            with self.preview_lock:
                self.latest_frame = frame.copy()
                self.latest_packet = FramePacket(
                    captured_monotonic_ms=time.monotonic_ns() // 1_000_000,
                    frame=frame,
                )

    def _inference_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.preview_lock:
                packet = self.latest_packet
                busy = self.inference_busy
                is_fresh = (
                    packet is not None and packet.captured_monotonic_ms > self.last_consumed_capture_ms
                )
                if not busy and is_fresh:
                    self.inference_busy = True
                    self.last_consumed_capture_ms = packet.captured_monotonic_ms
                    timestamp_ms = max(time.monotonic_ns() // 1_000_000, self.last_submitted_timestamp_ms + 1)
                    self.last_submitted_timestamp_ms = timestamp_ms
                    frame = packet.frame.copy()
                    frame_height, frame_width = frame.shape[:2]
                    self.pending_frame_sizes[timestamp_ms] = (frame_width, frame_height)
                else:
                    frame = None
                    timestamp_ms = None

            if frame is None or timestamp_ms is None:
                time.sleep(0.005)
                continue

            self.frame_events.put(
                PoseFrameEvent(
                    frame_timestamp_ms=timestamp_ms,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )
            )

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                self.landmarker.detect_async(mp_image, timestamp_ms)
            except Exception as exc:
                with self.preview_lock:
                    self.pending_frame_sizes.pop(timestamp_ms, None)
                    self.inference_busy = False
                self.log_events.put(f"Pose inference failed: {exc}")
                time.sleep(0.05)


def datetime_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
