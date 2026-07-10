"""Synchronized skeleton + Wi-Fi heatmap animation (layperson explainer).

Plays a short, clear segment of one capture as a video:
  - left  : the camera-measured pose drawn as a moving skeleton
  - right : the Wi-Fi CSI heatmap scrolling in real time (newest at the right)

So a non-expert can watch the skeleton move and see the Wi-Fi signal react at the
same instant. Output is a GIF (no ffmpeg needed).

Usage (from the ML dir):
    python scripts/tools/animate_dataset.py --action squart --dur 10
    python scripts/tools/animate_dataset.py --action stand_sidestep --start 20 --dur 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from explain_dataset import (  # noqa: E402
    project_dir,
    _ensure_src_path,
    collect_temporal_pose,
    _motion_signal,
    _smooth,
    _smooth_heatmap,
    _find_peaks,
    _action_motion,
    _select_story_window,
    EXPLAIN_ACTIONS,
    _heatmap_image,
    _mask_empty_rows,
    HEATMAP_CMAP,
    HEATMAP_EMPTY_COLOR,
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# bright, high-visibility color for the presentation time cursor / slider handle
CURSOR_COLOR = "#FC0C0CA7"
# black outline drawn under bright elements so they pop on any heatmap color
_CURSOR_GLOW = [pe.Stroke(linewidth=6, foreground="black"), pe.Normal()]

# skeleton edges over the 12 keypoints (x=2k, y=2k+1)
# 0 Lsh 1 Rsh 2 Lel 3 Rel 4 Lwr 5 Rwr 6 Lhip 7 Rhip 8 Lkne 9 Rkne 10 Lank 11 Rank
EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
    (6, 7), (0, 6), (1, 7),
    (6, 8), (8, 10), (7, 9), (9, 11),
]

# plain-language title / one-line takeaway for presentation slides
PRESENT_TITLES = {
    "squart": ("앉았다 일어서기", "몸이 아래로 내려갈 때마다 WiFi 신호가 크게 요동칩니다"),
    "stand_sidestep": ("옆으로 걷기", "한 걸음 옮길 때마다 WiFi 신호에 세로 줄무늬가 생깁니다"),
    "standlegupdown": ("다리 들었다 내리기", "다리를 들 때마다 WiFi 신호가 강하게 반응합니다"),
}


def _auto_window(motion, fps, dur):
    """Pick the start index whose dur-second window has the most regular reps."""
    wcols = int(dur * fps)
    if wcols >= len(motion):
        return 0
    best_i, best_score = 0, -1e9
    step = max(1, int(fps * 0.5))
    for i in range(0, len(motion) - wcols, step):
        seg = motion[i:i + wcols]
        peaks = _find_peaks(seg, min_dist=int(fps * 1.0), height=np.percentile(seg, 60))
        if len(peaks) < 3:
            continue
        gaps = np.diff(peaks)
        regularity = -np.std(gaps) / (np.mean(gaps) + 1e-9)   # higher = steadier
        score = len(peaks) + 2.0 * regularity
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def build_parser():
    p = argparse.ArgumentParser(description="Skeleton + Wi-Fi heatmap animation.")
    p.add_argument("--input-dir", type=Path, default=project_dir().parent.parent / "dataset" / "captures")
    p.add_argument("--output-dir", type=Path, default=project_dir() / "docs" / "figures" / "dataset_analysis")
    p.add_argument("--action", type=str, default="squart")
    p.add_argument("--pattern", type=str, default="sync_csi_pose*.csv")
    p.add_argument("--node-count", type=int, default=3)
    p.add_argument("--subcarrier-remap", type=str, default="esp32_htltf_ht40_above_nonstbc")
    p.add_argument("--file-index", type=int, default=0)
    p.add_argument("--start", type=float, default=-1.0, help="Segment start (s). -1 = auto-pick.")
    p.add_argument("--dur", type=float, default=8.0, help="Segment length (s).")
    p.add_argument("--window", type=float, default=6.0, help="Scrolling heatmap width (s).")
    p.add_argument("--play-fps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=100, help="Render resolution; higher = crisper but larger file.")
    p.add_argument("--present", action="store_true",
                   help="Presentation layout: skeleton + live scrolling heatmap on top, a full "
                        "overview heatmap with a moving cursor below. No titles (add them in the "
                        "slides). Saves as slide_<action>.gif.")
    p.add_argument("--overview-sec", type=float, default=30.0,
                   help="Length of the bottom overview window in --present mode.")
    p.add_argument("--play-sec", type=float, default=10.0,
                   help="How much time the top view actually animates (a dense sub-window of the "
                        "overview). Keeps the GIF small while the overview still shows the full span.")
    return p


def main():
    _ensure_src_path()
    args = build_parser().parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dir = args.input_dir.resolve() / f"captures_{args.action}"
    res = collect_temporal_pose(class_dir, args.pattern, args.node_count,
                                args.subcarrier_remap, args.file_index)
    if res is None:
        raise RuntimeError(f"No data for {args.action}")
    amp_seq, pose_seq, fps, fname = res
    print(f"[data] {args.action}: {amp_seq.shape[0]} frames ~{fps:.0f} Hz ({fname})")

    # full mean-removed heatmap [sc, T]; downsample subcarriers + smooth time
    # so the animated frames stay small and compress well as a GIF
    hm, sc_axis = _heatmap_image(amp_seq, args.subcarrier_remap, time_bins=amp_seq.shape[0], downsample=1)
    sw = max(1, int(fps * 0.12))
    if sw > 1:
        kernel = np.ones(sw) / sw
        hm = np.stack([np.convolve(row, kernel, mode="same") for row in hm])
    hm = _smooth_heatmap(hm, freq_win=5, time_win=max(5, int(fps * 0.18)))
    lim = float(np.percentile(np.abs(np.asarray(hm)), 99)) or 1.0
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad(HEATMAP_EMPTY_COLOR)
    row_mask = np.all(np.abs(np.asarray(hm)) < 1e-9, axis=1)
    motion = _smooth(_motion_signal(amp_seq), max(3, int(fps * 0.3)))

    def _yticks(ax):
        tick_count = min(5, len(sc_axis))
        tick_idx = np.linspace(0, len(sc_axis) - 1, tick_count, dtype=int)
        ax.set_yticks([float(sc_axis[i]) for i in tick_idx])
        ax.set_yticklabels([str(int(sc_axis[i])) for i in tick_idx], fontweight="bold")

    # ---------- presentation layout: skeleton + live scroll + full overview ----------
    if args.present:
        signal_raw, _, _, _ = _action_motion(args.action, pose_seq)
        ov_start, ov_end = _select_story_window(signal_raw, fps, story_sec=args.overview_sec)
        ov_dur = (ov_end - ov_start) / fps
        # playback = densest play_sec window inside the overview (keeps the GIF small)
        local = _auto_window(motion[ov_start:ov_end], fps, args.play_sec)
        play_start = ov_start + local
        play_end = min(ov_end, play_start + int(args.play_sec * fps))
        step = max(1, int(round(fps / args.play_fps)))
        frames = list(range(play_start, play_end, step))
        wcols = int(args.window * fps)
        print(f"[anim] overview {ov_start/fps:.1f}-{ov_end/fps:.1f}s ({ov_dur:.0f}s); "
              f"play {play_start/fps:.1f}-{play_end/fps:.1f}s, {len(frames)} frames")

        seg_pose = pose_seq[play_start:play_end]
        xs, ys = seg_pose[:, 0::2], seg_pose[:, 1::2]
        xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
        # generous, symmetric padding so joints never touch the panel edge
        mx = (xmax - xmin) * 0.5 + 0.04
        my = (ymax - ymin) * 0.12 + 0.04

        hm_over = hm[:, ov_start:ov_end]
        over_mask = np.repeat(row_mask[:, None], hm_over.shape[1], axis=1)

        fig = plt.figure(figsize=(13, 7.4))
        fig.patch.set_facecolor("white")       # opaque background so GIF frames fully
        fig.patch.set_alpha(1.0)               # overwrite (no accumulation of cursor/handle/xticks)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.3], height_ratios=[1.45, 1],
                              left=0.075, right=0.985, top=0.965, bottom=0.11,
                              hspace=0.45, wspace=0.16)
        ax_skel = fig.add_subplot(gs[0, 0])
        ax_live = fig.add_subplot(gs[0, 1])
        ax_over = fig.add_subplot(gs[1, :])

        # skeleton — centered, light panel, rounded bones, no ticks
        ax_skel.set_xlim(xmin - mx, xmax + mx)
        ax_skel.set_ylim(ymax + my, ymin - my)
        ax_skel.set_xticks([]); ax_skel.set_yticks([])
        ax_skel.set_aspect("equal")
        ax_skel.set_facecolor("#f5f6f8")
        for s in ax_skel.spines.values():
            s.set_color("#d0d3d9")
        bone_lines = [ax_skel.plot([], [], "-", color="#2b6cb0", lw=6,
                                   solid_capstyle="round", solid_joinstyle="round")[0] for _ in EDGES]
        joints = ax_skel.plot([], [], "o", color="#e53e3e", ms=12,
                              markeredgecolor="white", markeredgewidth=1.4)[0]

        def _style_heat(ax):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Time (s)", fontsize=22, fontweight="bold")
            ax.set_ylabel("Subcarrier Index", fontsize=22, fontweight="bold")
            _yticks(ax)
            ax.tick_params(labelsize=17, length=4, width=1.5)
            for lbl in ax.get_xticklabels():
                lbl.set_fontweight("bold")

        init = np.zeros((hm.shape[0], wcols))
        im_live = ax_live.imshow(
            np.ma.array(init, mask=np.repeat(row_mask[:, None], wcols, axis=1)),
            aspect="auto", origin="lower", cmap=cmap, interpolation="lanczos", resample=True,
            vmin=-lim, vmax=lim, extent=[-args.window, 0, float(sc_axis[0]), float(sc_axis[-1])])
        live_cursor = ax_live.axvline(0, color=CURSOR_COLOR, lw=4.0)
        live_cursor.set_path_effects(_CURSOR_GLOW)
        _style_heat(ax_live)

        im_over = ax_over.imshow(
            np.ma.array(hm_over, mask=over_mask),
            aspect="auto", origin="lower", cmap=cmap, interpolation="lanczos", resample=True,
            vmin=-lim, vmax=lim, extent=[0, ov_dur, float(sc_axis[0]), float(sc_axis[-1])])
        over_cursor = ax_over.axvline(0, color=CURSOR_COLOR, lw=5.0)
        over_cursor.set_path_effects(_CURSOR_GLOW)
        # colorbar on the right of the bottom overview heatmap
        cbar = fig.colorbar(im_over, ax=ax_over, fraction=0.025, pad=0.012)
        cbar.set_ticks([])                    # signed/arbitrary units -> gradient only
        cbar.outline.set_linewidth(1.4)
        # a slider "handle" that rides on top of the overview cursor
        over_handle = ax_over.plot(
            [0], [float(sc_axis[-1])], marker="v", color=CURSOR_COLOR, ms=20,
            markeredgecolor="black", markeredgewidth=1.6, clip_on=False, zorder=6)[0]
        ax_over.set_xlim(0, ov_dur)
        _style_heat(ax_over)

        def update(gi):
            pose = pose_seq[gi]
            px, py = pose[0::2], pose[1::2]
            for line, (a, b) in zip(bone_lines, EDGES):
                line.set_data([px[a], px[b]], [py[a], py[b]])
            joints.set_data(px, py)
            lo = max(0, gi - wcols)
            chunk = hm[:, lo:gi]
            if chunk.shape[1] < wcols:
                pad_cols = wcols - chunk.shape[1]
                if chunk.shape[1] > 0:
                    pad = np.repeat(chunk[:, :1], pad_cols, axis=1)
                    chunk = np.concatenate([pad, chunk], axis=1)
                else:
                    chunk = np.repeat(hm[:, :1], wcols, axis=1)
            im_live.set_data(np.ma.array(chunk, mask=np.repeat(row_mask[:, None], chunk.shape[1], axis=1)))
            t_rel = (gi - ov_start) / fps
            im_live.set_extent([t_rel - args.window, t_rel, float(sc_axis[0]), float(sc_axis[-1])])
            live_cursor.set_xdata([t_rel, t_rel])
            ax_live.set_xlim(t_rel - args.window, t_rel)
            over_cursor.set_xdata([t_rel, t_rel])
            over_handle.set_xdata([t_rel])
            return bone_lines + [joints, im_live, live_cursor, over_cursor, over_handle]

        anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.play_fps, blit=False)
        out_path = out_dir / f"slide_{args.action}.gif"
        anim.save(out_path, writer=PillowWriter(fps=args.play_fps), dpi=args.dpi,
                  savefig_kwargs={"facecolor": "white"})
        plt.close(fig)
        print(f"Done -> {out_path}")
        return 0

    # ---------- default (analysis) layout ----------
    # choose segment
    start_i = _auto_window(motion, fps, args.dur) if args.start < 0 else int(args.start * fps)
    seg_len = int(args.dur * fps)
    end_i = min(start_i + seg_len, amp_seq.shape[0])
    wcols = int(args.window * fps)
    step = max(1, int(round(fps / args.play_fps)))
    frames = list(range(start_i, end_i, step))
    print(f"[anim] segment {start_i/fps:.1f}-{end_i/fps:.1f}s, {len(frames)} frames")

    # skeleton axis limits from segment pose bbox
    seg_pose = pose_seq[start_i:end_i]
    xs, ys = seg_pose[:, 0::2], seg_pose[:, 1::2]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    mx = (xmax - xmin) * 0.35 + 0.02
    my = (ymax - ymin) * 0.15 + 0.02

    present = args.present
    lw = 5 if present else 3
    ms = 11 if present else 7
    title_fs = 16 if present else 12
    label_fs = 14 if present else 11

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.6) if present else (9.5, 4.3),
                                   gridspec_kw={"width_ratios": [1, 2]})

    # left: skeleton
    axL.set_title("실제 동작 (카메라)" if present else "① 실제 사람 움직임", fontsize=title_fs)
    axL.set_xlim(xmin - mx, xmax + mx)
    axL.set_ylim(ymax + my, ymin - my)          # invert y (image coords)
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_aspect("equal")
    bone_lines = [axL.plot([], [], "-", color="#1f77b4", lw=lw)[0] for _ in EDGES]
    joints = axL.plot([], [], "o", color="#d62728", ms=ms)[0]

    # right: scrolling heatmap
    axR.set_title("WiFi 신호 변화  (오른쪽 끝 = 지금)" if present else "② Wi-Fi CSI 분포", fontsize=title_fs)
    axR.set_xlabel("시간 (초)" if present else "Time (sec)", fontsize=label_fs)
    if present:
        axR.set_ylabel("WiFi 채널", fontsize=label_fs)
        axR.set_yticks([])
    else:
        axR.set_ylabel("Subcarrier index", fontsize=label_fs)
        tick_count = min(5, len(sc_axis))
        tick_idx = np.linspace(0, len(sc_axis) - 1, tick_count, dtype=int)
        axR.set_yticks([float(sc_axis[i]) for i in tick_idx])
        axR.set_yticklabels([str(int(sc_axis[i])) for i in tick_idx])
    init = np.zeros((hm.shape[0], wcols))
    im = axR.imshow(
        np.ma.array(init, mask=np.repeat(row_mask[:, None], init.shape[1], axis=1)),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        interpolation="lanczos",
        resample=True,
        vmin=-lim,
        vmax=lim,
        extent=[-args.window, 0, float(sc_axis[0]), float(sc_axis[-1])],
    )
    now_line = axR.axvline(0, color="w", lw=2.0 if present else 1.5)
    if present:
        title, caption = PRESENT_TITLES.get(args.action, (EXPLAIN_ACTIONS.get(args.action, (args.action,))[0], ""))
        fig.suptitle(title, fontsize=22, fontweight="bold", y=0.98)
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=14, color="0.2")
        time_txt = axR.text(0.99, 0.97, "", transform=axR.transAxes, ha="right", va="top",
                            fontsize=13, color="white",
                            bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.35))
        action_title = ""
    else:
        time_txt = fig.suptitle("", fontsize=14)
        action_title = EXPLAIN_ACTIONS.get(args.action, (args.action,))[0]

    def update(gi):
        pose = pose_seq[gi]
        px, py = pose[0::2], pose[1::2]
        for line, (a, b) in zip(bone_lines, EDGES):
            line.set_data([px[a], px[b]], [py[a], py[b]])
        joints.set_data(px, py)

        lo = max(0, gi - wcols)
        chunk = hm[:, lo:gi]
        if chunk.shape[1] < wcols:                # left-pad at the very start
            pad_cols = wcols - chunk.shape[1]
            if chunk.shape[1] > 0:
                pad = np.repeat(chunk[:, :1], pad_cols, axis=1)
                chunk = np.concatenate([pad, chunk], axis=1)
            else:
                chunk = np.repeat(hm[:, :1], wcols, axis=1)
        im.set_data(np.ma.array(chunk, mask=np.repeat(row_mask[:, None], chunk.shape[1], axis=1)))
        t = gi / fps
        im.set_extent([t - args.window, t, float(sc_axis[0]), float(sc_axis[-1])])
        now_line.set_xdata([t, t])
        axR.set_xlim(t - args.window, t)
        elapsed = t - start_i / fps
        time_txt.set_text(f"{elapsed:4.1f}초" if present else f"{action_title}   ·   {elapsed:4.1f}초")
        return bone_lines + [joints, im, now_line]

    if present:
        fig.subplots_adjust(left=0.03, right=0.99, top=0.87, bottom=0.17, wspace=0.08)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.play_fps, blit=False)
    prefix = "slide" if present else "animation"
    out_path = out_dir / f"{prefix}_{args.action}.gif"
    anim.save(out_path, writer=PillowWriter(fps=args.play_fps), dpi=max(args.dpi, 90 if present else 140))
    plt.close(fig)
    print(f"Done -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
