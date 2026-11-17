"""
Step 1 of moving the policy to the LEGO car: measure the real Double Motor.

Answers three questions the hardware-ready training task needs:
  1. Does the hub's IMU report yaw and gyro to Python? (never tested before)
  2. How fast does a leg turn (rad/s) for a given LEGO speed % ?
  3. How many times a second can we read sensors + send a command over Bluetooth?

Usage (laptop, from this folder, car on the floor with ~1 m of room ahead):
    ~/.venvs/me193/bin/python hw_calibrate.py

It walks forward at 30%, 60% and 100% for 3 s each, stopping between, then
saves lego_calibration.json next to this file. Paste the printout to Claude.
Pass: IMU line shows numbers (not nan), each speed shows a rad/s for both
legs, and the two legs are within ~15% of each other.
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # lelib.py next to this file (copied from ME193Sumo)

import legoeducation as le
from lelib import doubleMotor

SPEEDS = [30, 60, 100]   # LEGO speed % (forward)
CARD_SERIAL, CARD_COLOR = 994, "orange"   # our car's Connection Card: only connect to this hub
RUN_S = 3.0
LEFT, RIGHT = le.MOTOR_LEFT, le.MOTOR_RIGHT


def imu(dm):
    d = dm.imu_device
    return d.yaw, d.pitch, d.roll, d.gyroscopeX, d.gyroscopeY, d.gyroscopeZ


def main():
    print(f"Connecting to the Double Motor with card {CARD_COLOR} {CARD_SERIAL:04d}...")
    dm = doubleMotor()
    dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    time.sleep(1.0)  # let the first notifications arrive

    # 1. IMU
    dm.reset_heading()
    time.sleep(0.5)
    y, p, r, gx, gy, gz = imu(dm)
    imu_ok = not any(math.isnan(v) for v in (y, gx, gy, gz))
    print(f"\nIMU: yaw={y} pitch={p} roll={r} gyro=({gx}, {gy}, {gz})  -> {'OK' if imu_ok else 'NO DATA (nan)'}")

    # 3. loop rate: read everything + send one command, as the policy loop will
    n, t0 = 0, time.time()
    while time.time() - t0 < 2.0:
        _ = imu(dm), dm.motor[LEFT].position, dm.motor[RIGHT].position
        dm.movement_move_tank(0, 0)
        n += 1
    loop_hz = n / (time.time() - t0)
    print(f"Read + command loop: {loop_hz:.1f} Hz")

    # 2. speed % -> rad/s, from the cumulative encoder (degrees)
    results = []
    try:
        for pct in SPEEDS:
            l_cmd, r_cmd = pct, pct
            p0 = (dm.motor[LEFT].position, dm.motor[RIGHT].position)
            yaw0 = dm.imu_device.yaw
            t0 = time.time()
            reported = []
            dm.movement_move_tank(l_cmd, r_cmd)
            while time.time() - t0 < RUN_S:
                reported.append((dm.motor[LEFT].speed, dm.motor[RIGHT].speed))
                time.sleep(0.05)
            dt = time.time() - t0
            p1 = (dm.motor[LEFT].position, dm.motor[RIGHT].position)
            dm.movement_move_tank(0, 0)
            radps = [abs(b - a) * math.pi / 180 / dt for a, b in zip(p0, p1)]
            yaw_drift = dm.imu_device.yaw - yaw0
            rep = [sum(s[i] for s in reported[len(reported)//2:]) / max(1, len(reported) - len(reported)//2) for i in (0, 1)]
            print(f"{pct:3d}%: left {radps[0]:5.2f} rad/s  right {radps[1]:5.2f} rad/s  "
                  f"(hub reports speed {rep[0]:.0f}/{rep[1]:.0f})  yaw change {yaw_drift:+.0f} deg")
            results.append({"pct": pct, "left_radps": radps[0], "right_radps": radps[1],
                            "reported_speed": rep, "yaw_change": yaw_drift})
            time.sleep(1.5)
    finally:
        dm.movement_move_tank(0, 0)
        dm.stop()

    out = {"imu_ok": imu_ok, "loop_hz": loop_hz, "runs": results,
           "radps_per_pct": sum((r["left_radps"] + r["right_radps"]) / 2 / r["pct"] for r in results) / len(results)}
    with open(os.path.join(HERE, "lego_calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAbout {out['radps_per_pct']:.3f} rad/s per 1% speed  (100% = {100 * out['radps_per_pct']:.1f} rad/s)")
    print("Saved lego_calibration.json")


if __name__ == "__main__":
    main()
