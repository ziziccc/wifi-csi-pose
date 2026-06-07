from __future__ import annotations

import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

import serial
from serial.tools import list_ports


SERIAL_MAGIC = 0x35534943
SERIAL_HEADER = struct.Struct("<IBBHII")
STATUS_HEADER = struct.Struct("<8B7I6s2s")
STATUS_NODE = struct.Struct("<4B6sIII")
ACK_STRUCT = struct.Struct("<4BII64s")
CYCLE_HEADER = struct.Struct("<IIQQIBBBB")
CYCLE_SLOT = struct.Struct("<BBbBH")

FRAME_STATUS = 1
FRAME_CYCLE = 2
FRAME_ACK = 3

DEFAULT_BAUDRATE = 115_200
COMMAND_PREFIX = "CMD "
MAX_FRAME_PAYLOAD = 12_288
MAX_RX_BUFFER = 256 * 1024
READ_CHUNK_SIZE = 16_384


def checksum32(data: bytes) -> int:
    value = 0
    for byte in data:
        value = ((value << 5) - value + byte) & 0xFFFFFFFF
    return value


def mac_to_text(raw: bytes) -> str:
    return ":".join(f"{byte:02X}" for byte in raw)


def second_channel_text(value: int) -> str:
    if value == 1:
        return "above"
    if value == 2:
        return "below"
    return "none"


def decode_iq_pairs(payload: bytes) -> list[tuple[int, int]]:
    if not payload:
        return []
    values = struct.unpack(f"<{len(payload)}b", payload)
    return [(values[index], values[index + 1]) for index in range(0, len(values), 2)]


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


@dataclass
class ParsedRecord:
    host_time: float
    monotonic_ms: int
    uart_seq: int
    trigger_seq: int
    rx_index: int
    rssi: int
    csi_len: int
    record_bytes: int
    iq_pairs: list[tuple[int, int]]


@dataclass
class ParsedCycle:
    trigger_seq: int
    active_nodes: int
    received_nodes: int
    timeout_fired: bool
    timeout_packets: int
    records: list[ParsedRecord]


def parse_status(payload: bytes) -> dict[str, Any]:
    if len(payload) < STATUS_HEADER.size:
        raise ValueError("status payload too short")

    (
        mode,
        wifi_channel,
        second_channel,
        protocol_bitmap,
        connected_count,
        active_count,
        saved_count,
        node_entry_count,
        timeout_us,
        udp_slot_gap_us,
        generation,
        next_trigger_seq,
        uart_seq,
        trigger_sent_count,
        cycle_timeout_count,
        tx_mac,
        _,
    ) = STATUS_HEADER.unpack_from(payload, 0)

    nodes = []
    cursor = STATUS_HEADER.size
    expected_len = STATUS_HEADER.size + (node_entry_count * STATUS_NODE.size)
    if len(payload) < expected_len:
        raise ValueError(f"status payload truncated: expected>={expected_len} got={len(payload)}")

    for _ in range(node_entry_count):
        slot_index, saved_order, connect_order, flags, mac, last_seen_ms, rx_ok, rx_timeout = STATUS_NODE.unpack_from(
            payload, cursor
        )
        cursor += STATUS_NODE.size
        nodes.append(
            {
                "slot_index": slot_index,
                "saved_order": saved_order,
                "connect_order": connect_order,
                "flags": flags,
                "connected": bool(flags & 0x01),
                "saved": bool(flags & 0x02),
                "live": bool(flags & 0x04),
                "mac": mac_to_text(mac),
                "last_seen_ms": last_seen_ms,
                "rx_ok": rx_ok,
                "rx_timeout": rx_timeout,
            }
        )

    return {
        "mode": "running" if mode else "wait",
        "wifi_channel": wifi_channel,
        "second_channel": second_channel_text(second_channel),
        "protocol_bitmap": protocol_bitmap,
        "connected_count": connected_count,
        "active_count": active_count,
        "saved_count": saved_count,
        "timeout_us": timeout_us,
        "udp_slot_gap_us": udp_slot_gap_us,
        "generation": generation,
        "next_trigger_seq": next_trigger_seq,
        "uart_seq": uart_seq,
        "trigger_sent_count": trigger_sent_count,
        "cycle_timeout_count": cycle_timeout_count,
        "tx_mac": mac_to_text(tx_mac),
        "nodes": nodes,
    }


def parse_ack(payload: bytes) -> dict[str, Any]:
    ok, mode, wifi_channel, second_channel, timeout_us, udp_slot_gap_us, raw_message = ACK_STRUCT.unpack(payload)
    return {
        "ok": bool(ok),
        "mode": "running" if mode else "wait",
        "wifi_channel": wifi_channel,
        "second_channel": second_channel_text(second_channel),
        "timeout_us": timeout_us,
        "udp_slot_gap_us": udp_slot_gap_us,
        "message": raw_message.split(b"\x00", 1)[0].decode("ascii", errors="ignore"),
    }


def parse_compact_record(
    uart_seq: int,
    trigger_seq: int,
    rx_index: int,
    rssi: int,
    csi_bytes: bytes,
) -> ParsedRecord:
    if len(csi_bytes) & 1:
        csi_bytes = csi_bytes[:-1]

    return ParsedRecord(
        host_time=time.time(),
        monotonic_ms=time.monotonic_ns() // 1_000_000,
        uart_seq=uart_seq,
        trigger_seq=trigger_seq,
        rx_index=rx_index,
        rssi=rssi,
        csi_len=len(csi_bytes),
        record_bytes=len(csi_bytes),
        iq_pairs=decode_iq_pairs(csi_bytes),
    )


def parse_cycle(payload: bytes) -> ParsedCycle:
    if len(payload) < CYCLE_HEADER.size:
        raise ValueError("cycle payload too short")

    uart_seq, trigger_seq, _, _, _, active_nodes, received_nodes, timeout_fired, _ = CYCLE_HEADER.unpack_from(payload, 0)
    cursor = CYCLE_HEADER.size
    parsed: list[ParsedRecord] = []
    timeout_packets = max(active_nodes - received_nodes, 0)

    for _ in range(active_nodes):
        if len(payload) < cursor + CYCLE_SLOT.size:
            raise ValueError("cycle slot header truncated")
        rx_index, present, rssi, _, csi_len = CYCLE_SLOT.unpack_from(payload, cursor)
        cursor += CYCLE_SLOT.size
        if len(payload) < cursor + csi_len:
            raise ValueError("cycle csi payload truncated")
        csi_bytes = payload[cursor : cursor + csi_len]
        cursor += csi_len
        if not present:
            continue
        parsed.append(parse_compact_record(uart_seq, trigger_seq, rx_index, rssi, csi_bytes))

    return ParsedCycle(
        trigger_seq=trigger_seq,
        active_nodes=active_nodes,
        received_nodes=received_nodes,
        timeout_fired=bool(timeout_fired),
        timeout_packets=timeout_packets,
        records=parsed,
    )


class SerialReader(threading.Thread):
    def __init__(self, port: str, baudrate: int, event_queue: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.event_queue = event_queue
        self.stop_event = threading.Event()
        self.buffer = bytearray()
        self.serial_port: serial.Serial | None = None

    def stop(self) -> None:
        self.stop_event.set()
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except serial.SerialException:
                pass
        self.join(timeout=2.0)

    def send_line(self, line: str) -> None:
        if self.serial_port is None:
            return
        command = f"{COMMAND_PREFIX}{line.strip()}\n"
        try:
            self.serial_port.write(command.encode("ascii"))
        except serial.SerialTimeoutException:
            self.event_queue.put(("log", f"Serial write timeout: {line.strip()}"))
        except serial.SerialException as exc:
            self.event_queue.put(("log", f"Serial write failed: {exc}"))

    def run(self) -> None:
        try:
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=0.1,
            )
            self.serial_port.reset_input_buffer()
            self.event_queue.put(("log", f"Connected to {self.port} @ {self.baudrate}"))

            while not self.stop_event.is_set():
                chunk = self.serial_port.read(self.serial_port.in_waiting or READ_CHUNK_SIZE)
                if not chunk:
                    continue
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_RX_BUFFER:
                    self.event_queue.put(("log", f"RX buffer overflow, dropping {len(self.buffer)} bytes"))
                    del self.buffer[:-3]
                self._drain()
        except Exception as exc:
            self.event_queue.put(("log", f"Serial error: {exc}"))
        finally:
            self.event_queue.put(("closed", None))

    def _drain(self) -> None:
        magic_bytes = struct.pack("<I", SERIAL_MAGIC)

        while True:
            if len(self.buffer) < SERIAL_HEADER.size:
                return

            position = self.buffer.find(magic_bytes)
            if position < 0:
                del self.buffer[:-3]
                return
            if position > 0:
                del self.buffer[:position]
            if len(self.buffer) < SERIAL_HEADER.size:
                return

            magic, version, frame_type, payload_len, frame_seq, checksum = SERIAL_HEADER.unpack(
                self.buffer[: SERIAL_HEADER.size]
            )
            if (
                magic != SERIAL_MAGIC
                or version != 1
                or frame_type not in (FRAME_STATUS, FRAME_CYCLE, FRAME_ACK)
                or payload_len > MAX_FRAME_PAYLOAD
            ):
                del self.buffer[0]
                continue

            total_len = SERIAL_HEADER.size + payload_len
            if len(self.buffer) < total_len:
                return

            frame = bytes(self.buffer[:total_len])
            calc = checksum32(frame[: SERIAL_HEADER.size - 4] + frame[SERIAL_HEADER.size : total_len])
            if calc != checksum:
                self.event_queue.put(("log", f"Checksum mismatch frame_seq={frame_seq}"))
                next_position = self.buffer.find(magic_bytes, 1)
                if next_position >= 0:
                    del self.buffer[:next_position]
                else:
                    del self.buffer[0]
                continue

            del self.buffer[:total_len]

            payload = frame[SERIAL_HEADER.size :]
            try:
                if frame_type == FRAME_CYCLE:
                    cycle = parse_cycle(payload)
                    self.event_queue.put(("cycle", cycle))
                    for record in cycle.records:
                        self.event_queue.put(("record", record))
                elif frame_type == FRAME_STATUS:
                    self.event_queue.put(("status", parse_status(payload)))
                elif frame_type == FRAME_ACK:
                    self.event_queue.put(("ack", parse_ack(payload)))
                else:
                    self.event_queue.put(("log", f"Unknown frame type={frame_type} seq={frame_seq}"))
            except Exception as exc:
                self.event_queue.put(("log", f"Frame parse error type={frame_type} seq={frame_seq}: {exc}"))
