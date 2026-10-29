#!/usr/bin/env python3
"""Stage 3: validate recordings and plot their raw signals. Runs on the laptop or the Q.

    python scripts/validate_recording.py                      # newest file in data/raw/
    python scripts/validate_recording.py data/raw/ankle_sit_s01_2026-10-28T14-05.csv
    python scripts/validate_recording.py --all                # every file in data/raw/
    python scripts/validate_recording.py --plot               # also save <file>.png next to the CSV
    python scripts/validate_recording.py --plot --show        # …and open a plot window (laptop)

The activity and requested duration come from the filename and data/sessions.csv.
Exit code 0 = no FAILs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import config, imu, recording, validate


def requested_seconds(path):
    """Duration from sessions.csv if logged there (and not stopped early), else None (check skipped)."""
    return recording.expected_seconds(path.parent.parent / config.SESSIONS_CSV.name, path.name)


def plot(df, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")                    # no display needed (e.g. on the Q over SSH)
    import matplotlib.pyplot as plt

    t = (df["t_mcu_us"] - df["t_mcu_us"].iloc[0]) / 1e6
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for c in imu.ACCEL_COLS:
        axes[0].plot(t, df[c], lw=0.7, label=c)
    axes[0].set_ylabel("accel [g]")
    axes[1].plot(t, np.linalg.norm(df[imu.ACCEL_COLS].to_numpy(), axis=1), lw=0.7, color="k")
    axes[1].axhline(1.0, ls="--", lw=0.8, color="grey")
    axes[1].set_ylabel("|accel| [g]")
    for c in imu.GYRO_COLS:
        axes[2].plot(t, df[c], lw=0.7, label=c)
    axes[2].set_ylabel("gyro [°/s]")
    axes[3].plot(t.iloc[1:], np.diff(df["t_mcu_us"]) / 1000, lw=0.7, color="tab:purple")
    axes[3].axhline(1000 / config.SAMPLE_RATE_HZ, ls="--", lw=0.8, color="grey")
    axes[3].set_ylabel("Δt [ms]")
    axes[3].set_xlabel("time [s]")
    for ax in axes[[0, 2]]:
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(path.name)
    fig.tight_layout()
    out = path.with_suffix(".png")
    fig.savefig(out, dpi=110)
    print(f"  plot: {out}")
    if show:
        plt.show()
    plt.close(fig)


def validate_file(path, do_plot, show):
    print(f"\n{path.name}")
    try:
        activity, _ = recording.parse_filename(path.name)
    except ValueError:
        activity = None
    df = imu.load_csv(path)
    results = validate.run_all(df, activity, requested_seconds(path))
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  [{r.status}] {r.name:<{width}}  {r.detail}")
    if do_plot:
        plot(df, path, show)
    return all(r.ok for r in results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    files = args.files or sorted(args.dir.glob(f"{config.FILE_PREFIX}_*.csv"), key=lambda p: p.stat().st_mtime)
    if not args.files and not args.all:
        files = files[-1:]
    if not files:
        print(f"No recordings in {args.dir}. Record one with scripts/record.py, or copy them with scripts/fetch_data.py.")
        return 2
    ok = [validate_file(p, args.plot or args.show, args.show) for p in files]
    print(f"\n{sum(ok)}/{len(ok)} file(s) passed.")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
