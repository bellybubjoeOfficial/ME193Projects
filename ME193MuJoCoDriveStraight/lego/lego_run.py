"""
Run the trained Drive-Straight-Lego policy on the real LEGO car.

Usage (laptop, car on the floor pointing where you want it to go, ~1.5 m of room):
    ~/.venvs/me193/bin/python lego_run.py              # policy, 10 s
    ~/.venvs/me193/bin/python lego_run.py --open-loop  # no policy: both legs at the same fixed speed (baseline)
    ~/.venvs/me193/bin/python lego_run.py --seconds 15

Every 0.1 s (10 Hz, as in training) it reads the hub's yaw, builds the same 5 inputs the policy saw in simulation
[sin(yaw), cos(yaw), yaw rate, last action x2], runs the network with numpy, and sends one speed to each leg.
The heading at the start counts as "straight". A log (CSV) is written to ../results/lego_logs/ for the sim-vs-real comparison.
Ctrl-C stops the car.
"""
import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # lelib.py next to this file (copied from ME193Sumo)
from lelib import doubleMotor  # noqa: E402

RADPS_PER_PCT = 0.165   # measured with hw_calibrate.py: 100% = 15.9 rad/s
CARD_SERIAL, CARD_COLOR = 994, "orange"   # our car's Connection Card: only connect to this hub


class Policy:
    def __init__(self, path):
        p = np.load(path)
        self.mean, self.std = p["mean"], p["std"]
        self.layers = [(p["w0"], p["b0"]), (p["w1"], p["b1"]), (p["w2"], p["b2"])]
        self.order = [str(s) for s in p["action_order"]]
        self.max_leg_speed = float(p["max_leg_speed"])
        self.hz = int(p["control_hz"])

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if i < 2:
                x = np.where(x > 0, x, np.expm1(x))   # ELU
        return np.clip(x, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--open-loop", action="store_true", help="no policy: both legs at --speed %% the whole time")
    ap.add_argument("--speed", type=int, default=50, help="open-loop speed in %% (default 50)")
    ap.add_argument("--policy", default=os.path.join(HERE, "policy_lego.npz"))
    args = ap.parse_args()

    pol = None if args.open_loop else Policy(args.policy)
    hz = pol.hz if pol else 10
    dt = 1.0 / hz

    print(f"Connecting to the Double Motor with card {CARD_COLOR} {CARD_SERIAL:04d}...")
    dm = doubleMotor()
    dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    time.sleep(1.0)
    dm.reset_heading()
    time.sleep(0.5)

    runs = os.path.join(HERE, "..", "results", "lego_logs")
    os.makedirs(runs, exist_ok=True)
    log_path = os.path.join(runs, f"lego-{'openloop' if args.open_loop else 'policy'}-{datetime.now():%Y%m%d-%H%M%S}.csv")
    log = csv.writer(open(log_path, "w", newline=""))
    log.writerow(["t", "yaw_deg", "yaw_rate", "left_pct", "right_pct", "a_left", "a_right"])

    last_action = np.zeros(2)                 # in the policy's own order (pol.order)
    yaw_prev = None
    yaw_unwrapped = 0.0
    t0 = time.time()
    next_tick = t0
    step = 0
    print(f"{'open loop' if pol is None else 'policy'} for {args.seconds:.0f} s  (Ctrl-C stops)")
    print("   t    yaw(deg)  rate(rad/s)  left%  right%")
    try:
        while time.time() - t0 < args.seconds:
            now = time.time()
            yaw = math.radians(dm.imu_device.yaw / 10.0)        # hub: 0.1 deg, counter-clockwise positive
            if yaw_prev is None:
                yaw_prev = yaw
            d = (yaw - yaw_prev + math.pi) % (2 * math.pi) - math.pi   # unwrap across +-180 deg
            yaw_unwrapped += d
            rate = d / dt
            yaw_prev = yaw

            if pol is None:
                left_pct = right_pct = args.speed
                a = {"left_leg": 0.0, "right_leg": 0.0}
            else:
                obs = np.array([math.sin(yaw_unwrapped), math.cos(yaw_unwrapped), rate, *last_action])
                act = pol(obs)
                last_action = act
                a = dict(zip(pol.order, act))
                to_pct = pol.max_leg_speed / RADPS_PER_PCT
                left_pct = int(np.clip(round(a["left_leg"] * to_pct), -100, 100))
                right_pct = int(np.clip(round(a["right_leg"] * to_pct), -100, 100))
            dm.movement_move_tank(left_pct, right_pct)

            t = now - t0
            log.writerow([f"{t:.2f}", f"{math.degrees(yaw_unwrapped):.1f}", f"{rate:.3f}", left_pct, right_pct,
                          f"{a['left_leg']:.3f}", f"{a['right_leg']:.3f}"])
            if step % hz == 0:
                print(f"{t:5.1f}  {math.degrees(yaw_unwrapped):8.1f}  {rate:10.2f}  {left_pct:5d}  {right_pct:6d}")

            step += 1
            next_tick += dt
            time.sleep(max(0.0, next_tick - time.time()))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        dm.movement_move_tank(0, 0)
        dm.stop()
    print(f"final heading {math.degrees(yaw_unwrapped):+.1f} deg.  Log: {os.path.abspath(log_path)}")
    print("Measure with a tape: distance forward and sideways drift from the start line.")


if __name__ == "__main__":
    main()
