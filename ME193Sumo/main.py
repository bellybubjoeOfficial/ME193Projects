"""
main.py

Sumo wrestler control loop: read sensors → decision tree → motors.

Usage:
    python main.py                           # everything
    python main.py --no-color --no-whacker   # feet + Xbox only
    python main.py --no-unoq                 # without the UNO Q (no strikes; arm still homes to base)
Ctrl+C stops the motors and exits.
"""

import argparse
import math
import time

import legoeducation as le

import config
from drive import apply_inversion
from feet import FeetState, load_calibration, sign
from lelib import colorSensor, doubleMotor, singleMotor
from policy import Inputs, decide
from ring import EdgeGuard, is_tape
from sounds import SoundEffects
from unoq_link import UnoQLink
from whacker import (HOME, STRIKE, ArmGeometry, ArmMotion, ProximityTrigger, center_relative,
                     load_blind_calibration, load_center_abs, manual_move, visible_distance)
from xbox import ButtonEdges, XboxController


class ChangeSender:
    """Calls send(value) only when the value changes (or every RESEND_S as
    a keep-alive) so BLE isn't flooded."""

    def __init__(self, send):
        self._send = send
        self._last = None
        self._last_sent_at = 0.0

    def reset(self):
        """Forget the last value so the next call always sends."""
        self._last = None

    def __call__(self, value):
        now = time.monotonic()
        if value == self._last and now - self._last_sent_at < config.RESEND_S:
            return
        self._send(value)
        self._last = value
        self._last_sent_at = now


def wait_for_position(sm, timeout_s=3.0):
    """(relative, absolute) position once the motor has reported, else (None, None)."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        rel, abs_ = sm.motor.position, sm.motor.absolutePosition
        if not (math.isnan(rel) or math.isnan(abs_)):
            return rel, abs_
        time.sleep(0.05)
    return None, None


def parse_args():
    p = argparse.ArgumentParser(description="ME193 sumo wrestler")
    p.add_argument("--no-color", action="store_true", help="run without the color sensor")
    p.add_argument("--no-whacker", action="store_true", help="run without the whacker motor")
    p.add_argument("--no-unoq", action="store_true", help="run without the UNO Q distance sensor")
    return p.parse_args()


def main():
    args = parse_args()

    pad = XboxController()
    print("Waiting for Xbox controller (pair it in Bluetooth settings first)...")
    pad.connect()
    print(f"Controller: {pad.name}")

    print("Connecting to Double Motor (feet)...")
    dm = doubleMotor()
    dm.connect(card_serial=config.DRIVE_CARD_SERIAL, card_color=config.DRIVE_CARD_COLOR)

    cs = None
    if not args.no_color:
        print("Connecting to Color Sensor...")
        cs = colorSensor()
        cs.connect(card_serial=config.COLOR_CARD_SERIAL, card_color=config.COLOR_CARD_COLOR)

    sm = None
    if not args.no_whacker:
        print("Connecting to Single Motor (whacker)...")
        sm = singleMotor()
        sm.connect(card_serial=config.WHACKER_CARD_SERIAL, card_color=config.WHACKER_CARD_COLOR)
        sm.motor_set_end_state(le.MOTOR_END_STATE_HOLD)

    link = None
    if not args.no_unoq:
        print(f"Connecting to UNO Q broker {config.MQTT_BROKER}...")
        link = UnoQLink()
        link.connect()

    geometry = None
    if sm:
        center_abs = load_center_abs()
        rel, abs_ = wait_for_position(sm)
        if center_abs is None or rel is None:
            print("!! whacker not calibrated (or no position) -- no base/strike, and RT/LT "
                  "have NO end-point limits. Run tests/hw_whacker_calibrate.py first.")
        else:
            geometry = ArmGeometry(center=center_relative(rel, abs_, center_abs))
            print(f"Whacker: center abs {center_abs:.0f}°, base {geometry.base}, "
                  f"strike end {geometry.strike_end} (relative counter, now {rel})")

    blind_arcs = load_blind_calibration()
    if link and sm and blind_arcs is None:
        print("!! whacker_calibration.json not found -- the whacker may trigger itself. "
              "Run tests/hw_whacker_calibrate.py first.")
        blind_arcs = []

    arcs = load_calibration()
    if arcs is None:
        print("!! feet_calibration.json not found -- feet won't settle. "
              "Run tests/hw_feet_calibrate.py first.")
    print("Ready. Left stick drives, B brakes, RT/LT swing the whacker, Y = sad, A = victory."
          + (f" Strike when something is within {config.WHACK_TRIGGER_MM} mm."
             if link and geometry else ""))

    def send_feet(cmd):
        if cmd == (0, 0):
            dm.stop()
        else:
            dm.movement_move_tank(*cmd)

    def send_whack(speed):
        if speed == 0:
            sm.stop()
        else:
            sm.run(speed)

    def send_whack_target(move):
        # Hub-side position control: decelerates into the target and stops
        # there by itself, so the arm can't overshoot an end point.
        target, speed = move
        sm.motor_run_to_relative_position(target, speed=speed, blocking=False)

    feet_out = ChangeSender(send_feet)
    whack_out = ChangeSender(send_whack) if sm else None
    whack_target_out = ChangeSender(send_whack_target) if sm else None
    sfx = SoundEffects()
    buttons = ButtonEdges()
    guard = EdgeGuard()
    prox = ProximityTrigger()
    motion = ArmMotion(geometry) if geometry else None
    last_manual = -math.inf
    period = 1.0 / config.LOOP_HZ
    last_reason = None
    left_dir = right_dir = 0
    try:
        while True:
            tick = time.monotonic()
            tape_hit = guard.update(is_tape(cs.detect_color()), tick) if cs else False

            feet = None
            left_pos = dm.motor[le.MOTOR_LEFT].absolutePosition
            right_pos = dm.motor[le.MOTOR_RIGHT].absolutePosition
            # Positions are NaN until the hub's first motor notification.
            if arcs is not None and not math.isnan(left_pos) and not math.isnan(right_pos):
                feet = FeetState(
                    left_pos=left_pos,
                    right_pos=right_pos,
                    left_arc=arcs[0], right_arc=arcs[1],
                    left_dir=left_dir, right_dir=right_dir,
                )

            # Whacker inputs: strikes need the UNO Q + calibrated arm geometry.
            opponent_entered = False
            whack_pos = sm.motor.position if sm else math.nan
            known = geometry is not None and not math.isnan(whack_pos)
            if link and known:
                # Ignore the sensor while the whacker itself is in front of it.
                mm = visible_distance(link.distance_mm(), sm.motor.absolutePosition, blind_arcs)
                opponent_entered = prox.update(mm)
            arm_target = motion.step(whack_pos, tick) if motion and motion.active else None

            state = pad.read()
            action = decide(Inputs(
                pad=state,
                pressed=buttons.update(state),
                tape_hit=tape_hit,
                blocked_dir=guard.blocked_direction(tick),
                feet=feet,
                opponent_entered=opponent_entered,
                arm_phase=motion.phase if motion else None,
                arm_target=arm_target,
                arm_at_base=geometry.at_base(whack_pos) if known else None,
                manual_idle_s=tick - last_manual,
            ))

            if action.arm_command == "cancel":
                motion.cancel()
            elif action.arm_command in (STRIKE, HOME):
                motion.start(action.arm_command, tick)
                print(f"!! whacker: {action.arm_command}")
            if action.whack_mode == "manual":
                last_manual = tick

            feet_out(apply_inversion(action.left, action.right))
            if sm:
                move = None
                if action.whack_mode in (STRIKE, HOME):
                    move = (action.whack_target, config.WHACKER_SPEED)
                elif action.whack_mode == "manual" and geometry:
                    move = manual_move(action.whack, geometry)   # limited to base..strike end
                if move:
                    # Don't send speeds/stops now: they'd abort the hub's position move.
                    whack_target_out(move)
                    whack_out.reset()
                else:
                    whack_target_out.reset()
                    whack_out(action.whack)
            left_dir = sign(action.left) or left_dir
            right_dir = sign(action.right) or right_dir

            if action.sound:
                sfx.play(action.sound)
            if action.alert:
                sfx.play_edge()
                pad.rumble(config.RUMBLE_STRENGTH, config.RUMBLE_MS)
                print("!! tape")
            if action.reason != last_reason:
                print(f"[{action.reason}]")
                last_reason = action.reason
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        dm.stop()
        if sm:
            sm.stop()
        if link:
            link.close()
        pad.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
