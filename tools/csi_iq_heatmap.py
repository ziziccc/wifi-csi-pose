from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_iq_pairs(raw: str) -> list[tuple[int, int]]:
    if not raw:
        return []
    pairs = ast.literal_eval(raw)
    return [(int(i), int(q)) for i, q in pairs]


def amplitude_row(raw_pairs: str) -> np.ndarray:
    pairs = parse_iq_pairs(raw_pairs)
    return np.asarray([math.hypot(i, q) for i, q in pairs], dtype=np.float32)


def load_csv_rows(csv_path: Path, rx_index: int | None) -> dict[int, dict[int, np.ndarray]]:
    rows_by_rx: dict[int, dict[int, np.ndarray]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                current_rx = int(row["rx_index"])
                trigger_seq = int(row["trigger_seq"])
            except (KeyError, TypeError, ValueError):
                continue
            if rx_index is not None and current_rx != rx_index:
                continue
            try:
                amplitudes = amplitude_row(row.get("iq_pairs", ""))
            except (SyntaxError, ValueError, TypeError):
                continue
            if amplitudes.size == 0:
                continue
            rows_by_rx.setdefault(current_rx, {}).setdefault(trigger_seq, amplitudes)
    return rows_by_rx


def trigger_axis(rows_by_rx: dict[int, dict[int, np.ndarray]], max_rows: int | None) -> list[int]:
    trigger_values = sorted({trigger for rows in rows_by_rx.values() for trigger in rows})
    if not trigger_values:
        return []
    start = trigger_values[0]
    end = trigger_values[-1]
    if max_rows is not None:
        end = min(end, start + max_rows - 1)
    return list(range(start, end + 1))


def aligned_matrix(rows: dict[int, np.ndarray], triggers: list[int]) -> np.ndarray:
    if not rows or not triggers:
        return np.zeros((0, 0), dtype=np.float32)
    width = max(len(row) for row in rows.values())
    trigger_to_col = {trigger: index for index, trigger in enumerate(triggers)}
    matrix = np.full((width, len(triggers)), np.nan, dtype=np.float32)
    for trigger, row in rows.items():
        col = trigger_to_col.get(trigger)
        if col is None:
            continue
        matrix[: len(row), col] = row
    return matrix


def build_heatmap(
    csv_path: Path,
    output_path: Path,
    rx_index: int | None,
    max_rows: int | None,
    log_scale: bool,
    dpi: int,
) -> None:
    rows_by_rx = load_csv_rows(csv_path, rx_index=rx_index)
    if not rows_by_rx:
        raise SystemExit("No valid iq_pairs rows found.")

    if log_scale:
        rows_by_rx = {
            rx: {trigger: np.log1p(row) for trigger, row in rows.items()}
            for rx, rows in rows_by_rx.items()
        }

    rx_keys = sorted(rows_by_rx)
    triggers = trigger_axis(rows_by_rx, max_rows=max_rows)
    if not triggers:
        raise SystemExit("No valid trigger_seq values found.")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")

    fig_height = max(3.0, 2.4 * len(rx_keys))
    fig, axes = plt.subplots(len(rx_keys), 1, figsize=(14, fig_height), squeeze=False)
    axes_flat = axes[:, 0]
    image = None

    for axis, rx in zip(axes_flat, rx_keys):
        matrix = aligned_matrix(rows_by_rx[rx], triggers)
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, origin="lower")
        present_count = sum(1 for trigger in triggers if trigger in rows_by_rx[rx])
        present_pct = (present_count / max(len(triggers), 1)) * 100.0
        axis.set_title(
            f"RX {rx} CSI IQ amplitude "
            f"({matrix.shape[0]} CSI pairs x {len(triggers)} trigger_seq columns, "
            f"present {present_count}/{len(triggers)} = {present_pct:.1f}%)"
        )
        axis.set_xlabel("trigger_seq")
        axis.set_ylabel("CSI pair index")
        tick_count = min(6, len(triggers))
        if tick_count > 1:
            tick_positions = np.linspace(0, len(triggers) - 1, tick_count, dtype=int)
            axis.set_xticks(tick_positions)
            axis.set_xticklabels([str(triggers[index]) for index in tick_positions], rotation=0)

    if image is not None:
        label = "log(1 + amplitude)" if log_scale else "amplitude sqrt(I^2 + Q^2)"
        colorbar_axis = fig.add_axes((0.93, 0.12, 0.018, 0.76))
        fig.colorbar(image, cax=colorbar_axis, label=label)

    fig.suptitle(csv_path.name, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.93, bottom=0.08, hspace=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render CSI IQ pair amplitude heatmaps from synchronized CSV files.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="+",
        help="Input CSV path(s). Each CSV is rendered to a PNG with the same file name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG files. Default writes next to each input CSV.",
    )
    parser.add_argument("--rx-index", type=int, default=None, help="Only render one rx_index. Default renders all RXs.")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum packets per RX to render.")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use raw amplitude color scale. Default is log(1 + amplitude).",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG DPI.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for csv_path in args.csv:
        output_path = (
            args.output_dir / csv_path.with_suffix(".png").name
            if args.output_dir is not None
            else csv_path.with_suffix(".png")
        )
        build_heatmap(
            csv_path=csv_path,
            output_path=output_path,
            rx_index=args.rx_index,
            max_rows=args.max_rows,
            log_scale=not args.linear,
            dpi=args.dpi,
        )
        print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
