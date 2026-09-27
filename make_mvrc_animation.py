"""
Ghost-race animation for a batch of MVRC simulations.

Every car starts on the start/finish line at t = 0 and runs its own simulated lap, so
the field spreads out over the lap exactly in proportion to the simulated pace. A live
leaderboard on the right shows the running order and the time gap to the leader.

Reads the per-car .npz telemetry dumps written by run_mvrc_batch.py, so the animation
can be re-rendered (different fps, speed, colours) without re-running the simulations.

Usage:
    conda run -n lts313 python make_mvrc_animation.py <telemetry-dir> [-o <out.mp4>]
                                                      [--fps 30] [--speed 1.0] [--tail 2.0]

Output format follows the extension: .mp4 (ffmpeg) or .gif (pillow).
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")  # batch script: render to file, never to a display

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Shared look with the telemetry plots.
_PAGE_BACKGROUND = "#0e1117"
_FG = "#e6e6e6"
_MUTED = "#9aa0aa"
_TRACK_COLOR = "#2b303b"

# Distinct, saturated colours that stay legible on the dark ground. Cars beyond this
# many entries wrap around; the leaderboard still separates them by name.
_CAR_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#a9a9a9",
]


class CarRun:
    """One simulated lap, resampled onto the animation's common time grid."""

    def __init__(self, path: str):
        data = np.load(path, allow_pickle=False)
        self.team = str(data["team"])
        self.track = str(data["track"])
        self.lap_time = float(data["lap_time"])
        self.time = data["time"]
        self.distance = data["distance"]
        self.track_x = data["track_x"]
        self.track_y = data["track_y"]
        self.velocity_kmh = data["velocity_kmh"]
        self.total_distance = float(self.distance[-1])

    def distance_at(self, t: np.ndarray) -> np.ndarray:
        """Distance covered at each time. Clipped at the lap distance once the car has finished."""
        return np.interp(t, self.time, self.distance,
                         left=0.0, right=self.total_distance)

    def position_at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Track position at a given distance along the lap."""
        return (np.interp(s, self.distance, self.track_x),
                np.interp(s, self.distance, self.track_y))

    def speed_at(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self.distance, self.velocity_kmh)

    def time_to_reach(self, s: float) -> float:
        """When this car reaches distance s. Used for the time gap to the leader."""
        return float(np.interp(s, self.distance, self.time))


def load_runs(telemetry_dir: str) -> list[CarRun]:
    paths = sorted(glob.glob(os.path.join(telemetry_dir, "*.npz")))
    if not paths:
        raise SystemExit(
            f"no .npz telemetry found in {telemetry_dir} -- run run_mvrc_batch.py first"
        )
    runs = [CarRun(p) for p in paths]
    runs.sort(key=lambda r: r.lap_time)  # fastest first, so colours follow the result order
    return runs


def format_lap_time(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:06.3f}"


def build_animation(runs: list[CarRun], fps: int, speed: float, tail_s: float,
                    hold_s: float = 2.0):
    """Build the ghost-race figure and its FuncAnimation."""
    slowest = max(r.lap_time for r in runs)
    # One frame every 'speed / fps' seconds of simulated time, plus a hold on the final
    # frame so the finishing order stays readable at the end of the clip.
    dt = speed / fps
    sim_t = np.arange(0.0, slowest + dt, dt)
    n_hold = int(round(hold_s * fps))

    # Precompute every car's distance/position for every frame: a few thousand frames
    # times a handful of cars is small, and it keeps the per-frame callback trivial.
    dists = np.array([r.distance_at(sim_t) for r in runs])            # (n_cars, n_frames)
    xs = np.empty_like(dists)
    ys = np.empty_like(dists)
    for i, r in enumerate(runs):
        xs[i], ys[i] = r.position_at(dists[i])
    speeds = np.array([r.speed_at(dists[i]) for i, r in enumerate(runs)])

    colors = [_CAR_COLORS[i % len(_CAR_COLORS)] for i in range(len(runs))]

    # --- Figure -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), facecolor=_PAGE_BACKGROUND)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.02,
                          left=0.02, right=0.98, top=0.90, bottom=0.03)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_board = fig.add_subplot(gs[0, 1])

    for ax in (ax_map, ax_board):
        ax.set_facecolor(_PAGE_BACKGROUND)
        ax.axis("off")

    # Track outline -- every car runs the same raceline, so one outline serves all.
    ax_map.plot(runs[0].track_x, runs[0].track_y, color=_TRACK_COLOR, lw=14,
                solid_capstyle="round", zorder=1)
    ax_map.plot(runs[0].track_x[0], runs[0].track_y[0], "s", color="white",
                markeredgecolor="black", ms=10, zorder=2)
    ax_map.set_aspect("equal")
    ax_map.autoscale()
    ax_map.margins(0.04)

    # Trailing tails give a sense of direction and speed without cluttering the map.
    tail_pts = max(1, int(round(tail_s / dt)))
    tails = [ax_map.plot([], [], color=c, lw=3, alpha=0.45, solid_capstyle="round",
                         zorder=3)[0] for c in colors]
    markers = [ax_map.plot([], [], "o", color=c, ms=13, markeredgecolor="black",
                           markeredgewidth=1.0, zorder=4)[0] for c in colors]

    fig.suptitle(f"MVRC 2026 — {runs[0].track}", color=_FG, fontsize=20,
                 fontweight="bold", y=0.965)
    fig.text(0.98, 0.925, "Ghost race — every car starts on the line at t = 0",
             color=_MUTED, fontsize=11, ha="right")

    # The clock has to live inside an axis: blitting only repaints the axes regions it
    # is handed, so a figure-level text would be captured in the static background once
    # and never updated.
    clock = ax_map.text(0.01, 0.99, "", transform=ax_map.transAxes, color=_FG,
                        fontsize=17, family="monospace", va="top", zorder=5)

    # --- Leaderboard ------------------------------------------------------------------
    ax_board.set_xlim(0, 1)
    ax_board.set_ylim(0, 1)
    row_h = 0.055
    top = 0.95
    ax_board.text(0.0, top + 0.03, "ORDER", color=_MUTED, fontsize=11,
                  family="monospace", va="bottom")
    rows = [ax_board.text(0.0, top - i * row_h, "", color=c, fontsize=13,
                          family="monospace", va="top")
            for i, c in enumerate(colors)]

    artists = tails + markers + rows + [clock]  # everything blitting has to repaint

    def update(frame):
        f = min(frame, len(sim_t) - 1)  # the hold frames repeat the last real frame
        t = sim_t[f]

        for i in range(len(runs)):
            markers[i].set_data([xs[i, f]], [ys[i, f]])
            lo = max(0, f - tail_pts)
            tails[i].set_data(xs[i, lo:f + 1], ys[i, lo:f + 1])
            # Cars that have taken the flag all sit on the same spot; fading them keeps
            # the still-running cars readable instead of buried under the pile.
            done = t >= runs[i].lap_time
            markers[i].set_alpha(0.2 if done else 1.0)
            tails[i].set_alpha(0.0 if done else 0.45)

        # Running order is by distance covered, i.e. who is physically ahead on the map.
        # The gap is how long ago the leader passed this car's current position -- that
        # measure increases monotonically with the distance deficit, so the numbers can
        # never contradict the order shown. (Asking instead when this car will reach the
        # leader's position depends on its own speed at that instant and can invert.)
        order = np.argsort(-dists[:, f])
        leader = runs[order[0]]
        for pos, car in enumerate(order):
            r = runs[car]
            finished = t >= r.lap_time
            if finished:
                detail = f"FIN {format_lap_time(r.lap_time)}"
            elif pos == 0:
                detail = f"{speeds[car, f]:5.0f} km/h"
            else:
                gap = t - leader.time_to_reach(dists[car, f])
                detail = f"+{gap:5.2f}s"
            rows[pos].set_text(f"{pos + 1:2d}  {r.team[:18]:<18} {detail}")
            rows[pos].set_color(colors[car])
            rows[pos].set_alpha(0.55 if finished else 1.0)

        clock.set_text(f"{format_lap_time(t)}")
        return artists

    n_frames = len(sim_t) + n_hold
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=True,
    )
    return fig, anim, n_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("telemetry_dir",
                        help="directory holding the per-car .npz dumps from run_mvrc_batch.py")
    parser.add_argument("-o", "--output", default=None,
                        help="output video (default: <telemetry-dir>/../ghost_race.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default: 30)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed relative to real time (default: 1.0)")
    parser.add_argument("--tail", type=float, default=2.0,
                        help="length of each car's trail in simulated seconds (default: 2.0)")
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.telemetry_dir)), "ghost_race.mp4"
    )

    runs = load_runs(args.telemetry_dir)
    print(f"{len(runs)} cars at {runs[0].track}:")
    for i, r in enumerate(runs, start=1):
        print(f"  {i:2d}. {r.team:<20} {format_lap_time(r.lap_time)}")

    fig, anim, n_frames = build_animation(runs, fps=args.fps, speed=args.speed,
                                          tail_s=args.tail)

    writer = "pillow" if out_path.lower().endswith(".gif") else "ffmpeg"
    print(f"\nRendering {n_frames} frames at {args.fps} fps with {writer}...")
    anim.save(out_path, writer=writer, fps=args.fps,
              savefig_kwargs={"facecolor": _PAGE_BACKGROUND})
    plt.close(fig)

    print(f"Animation written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
