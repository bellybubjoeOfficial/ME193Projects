"""
No-policy simulation that reproduces the drift measured on the real LEGO car.

Both legs are commanded 50% (8.25 rad/s) for 10 s, like `lego_run.py --open-loop`. The simulated robot is perfectly
symmetric and walks straight; the real car turned ~115 deg right.

What we tried (2026-11-15):
  - different grip per leg (right-leg friction 0.1..1.0): at most a few degrees, wrong direction -> not the cause
  - legs starting out of phase (0..315 deg): up to ~40 deg, not a steady turn -> not the main cause
  - one leg doing more work per turn: reproduces the steady right turn
The real legs spin at the same rate (hw_calibrate.py), so the fitted ratio is an *effective stride* difference: a
single number standing in for the real car's unevenness (beam length or mounting angle, slip, weight on one side).
This script sweeps that ratio (left leg's effective speed / right leg's), picks the one whose heading curve best
matches the real log, and films it.

Usage (laptop, from this folder):
    MUJOCO_GL=cgl ~/.venvs/onshape/bin/python sim_openloop_drift.py ../results/lego_logs/lego-openloop-<time>.csv
Writes ../results/videos/sim_openloop_drift.mp4 and prints the fit.
"""
import csv
import math
import os
import sys

import imageio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "..", "onshape", "scene.xml")
OUT_DIR = os.path.join(HERE, "..", "results", "videos")
SPEED = 0.165 * 50          # 50% on the real motor, rad/s
SECONDS = 10.0
SWEEP = np.round(np.arange(1.00, 1.401, 0.02), 2)  # left/right effective-stride ratios to try (left > right = turns right)


def build():
    m = mujoco.MjSpec.from_file(SCENE).compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    d.qpos[2] += 0.01 - (d.geom_xpos[:, 2] - m.geom_rbound).min()
    chassis = m.body("part_1").id
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == chassis:
            m.geom_friction[g, 0] = 0.01           # chassis slides (same as drive.py / training)
    return m, d, chassis


def yaw_of(d):
    w, x, y, z = d.qpos[3:7]
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def simulate(ratio, times, film=False):
    """Heading (deg, unwrapped) at each of `times`; optionally frames at 30 fps and a per-second table."""
    m, d, chassis = build()
    L, R = m.actuator("left_leg").id, m.actuator("right_leg").id
    ren = cam = None
    if film:
        ren = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, chassis
        cam.distance, cam.azimuth, cam.elevation = 0.6, 140, -35
    frames, table, out = [], [], []
    yaw_u, prev = 0.0, 0.0
    ti, n_frame = 0, 0
    while d.time < SECONDS:
        d.ctrl[L] += SPEED * ratio * m.opt.timestep
        d.ctrl[R] += SPEED * m.opt.timestep
        mujoco.mj_step(m, d)
        y = yaw_of(d)
        yaw_u += (y - prev + 180) % 360 - 180
        prev = y
        while ti < len(times) and d.time >= times[ti]:
            out.append(yaw_u)
            ti += 1
        if film and d.time >= n_frame / 30:
            ren.update_scene(d, camera=cam)
            frames.append(ren.render())
            n_frame += 1
            if n_frame % 30 == 0:
                table.append((n_frame // 30, d.qpos[1], d.qpos[0], yaw_u))
    while len(out) < len(times):
        out.append(yaw_u)
    return np.array(out), frames, table


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: sim_openloop_drift.py <../results/lego_logs/lego-openloop-....csv>")
    rows = list(csv.DictReader(open(sys.argv[1])))
    t_real = np.array([float(r["t"]) for r in rows])
    yaw_real = np.array([float(r["yaw_deg"]) for r in rows])

    print("left/right stride ratio   final heading (deg)   RMS error vs real (deg)")
    best = None
    for k in SWEEP:
        yaw_sim, _, _ = simulate(k, t_real)
        rms = float(np.sqrt(np.mean((yaw_sim - yaw_real) ** 2)))
        print(f"        {k:4.2f}                {yaw_sim[-1]:+8.1f}              {rms:7.1f}")
        if best is None or rms < best[1]:
            best = (k, rms)
    k, rms = best
    print(f"\nbest: left leg effectively {100 * (k - 1):.0f}% stronger than right, RMS heading error {rms:.1f} deg; "
          f"real final heading {yaw_real[-1]:+.1f} deg")

    yaw_sim, frames, table = simulate(k, t_real, film=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "sim_openloop_drift.mp4")
    imageio.mimwrite(out, frames, fps=30)
    real_at = {int(round(t)): y for t, y in zip(t_real, yaw_real) if abs(t - round(t)) < 0.06}
    print("\n  second  forward(m)  sideways(m)  heading sim (deg)  heading real (deg)")
    for s, fwd, side, yaw in table:
        r = real_at.get(s)
        print(f"  {s:5d}  {fwd:9.2f}  {side:10.2f}  {yaw:15.1f}  {('%15.1f' % r) if r is not None else '              -'}")
    print(f"\nwrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
