"""
wacker_pwm_test.py

Standalone bench-test tool for the torsion-spring "wacker" hammer
mechanism (a motor winds a torsion spring back, holds it there, then
releases it so it snaps forward into the ping pong ball). This has
nothing to do with the IMU/game-loop code elsewhere in this project --
it exists purely to find, on a Lego Education Double Motor hub with the
hammer plugged into one side (or both):

  - What duty cycle (= PWM level, -100 to 100) is needed to *wind* the
    spring back at a reasonable speed.
  - What duty cycle is needed to *hold* the spring wound without it
    creeping back -- the spring's torque fights the motor's holding
    torque, so watch `position` while holding: if it drifts back
    toward 0, that duty cycle is too weak to hold.
  - How the three built-in motor end-states (hold/brake/coast) behave
    on release, as an alternative to holding with a duty cycle
    yourself.

Connects to whichever Double Motor hub advertises first (no connection
card needed), the same "closest bluetooth signal" connection style
already used in real_racquet_input.py.

With --motor both: the two sides are driven with the SAME commanded
duty cycle/direction mirrored (right side negated, see
MIRROR_RIGHT_SIDE) so they spin the same real-world direction despite
being mounted facing opposite ways on the hub, and both commands are
fired back-to-back with no artificial delay between them (the first
goes out non-blocking, the second blocks as usual) so there's no lag
between the two sides starting.

Usage:
    python wacker_pwm_test.py                    # hammer on the left side (default)
    python wacker_pwm_test.py --motor right       # hammer on the right side
    python wacker_pwm_test.py --motor both        # hammer driven by both sides together
    python wacker_pwm_test.py --card-serial 1234  # pin to a specific card instead

Once connected, type commands at the prompt (telemetry -- position,
absolute position, measured power, measured speed, for whichever
side(s) --motor selected -- prints after every command):

    <number>          Set duty cycle to this value now (-100 to 100).
                      Positive = clockwise, negative = counter-clockwise.
                      This is a raw PWM level: it keeps driving the
                      motor continuously until you send another command.
    0                 Shorthand for duty cycle 0 (motor goes slack --
                      NOT the same as holding; the spring can push it).
    watch [seconds]   Keep printing live telemetry with no new command
                      sent, so you can watch `position` for creep while
                      whatever duty cycle is currently set keeps
                      running. Omit seconds to watch until Ctrl+C.
    wind <deg> [speed]
                      Blocking: winds the spring back by <deg> degrees
                      at <speed>% (closed-loop speed control, default
                      40%), then stops. Use this for a repeatable
                      starting point before testing hold duty cycles.
                      <deg> can be negative to wind the other way.
    zero              Reset the position counter to 0 -- do this right
                      before a `wind` so `position` reads "how far
                      wound" from that point.
    hold              motor_stop() with end-state HOLD (the hub's own
                      active hold, not a duty cycle you chose).
    brake             motor_stop() with end-state BRAKE (shorts the
                      motor leads -- resists being turned, but doesn't
                      actively drive back to position like HOLD).
    coast / release   motor_stop() with end-state COAST (motor goes
                      fully slack) -- this is the "let the spring snap
                      the hammer forward" release.
    q / quit          Stop the motor (coast) and disconnect.
"""

import argparse
import time

import legoeducation as le

from lelib import doubleMotor

DEFAULT_WIND_SPEED = 40

# Set True if the two physical motors are mounted facing opposite ways
# (common when a single Double Motor hub drives two symmetric outputs)
# so that the SAME commanded duty cycle/direction spins them in
# opposite real-world directions. With this on, --motor both negates
# the right side's command so both actually spin the same way. Flip to
# False if they're still opposite, or if they start matching but in the
# wrong shared direction, flip the sign you type instead.
MIRROR_RIGHT_SIDE = True

MOTOR_SIDE_BY_NAME = {
    "left": le.MOTOR_LEFT,
    "right": le.MOTOR_RIGHT,
    "both": le.MOTOR_BOTH,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Bench-test PWM levels for the wacker hammer motor")
    parser.add_argument("--motor", choices=["left", "right", "both"], default="left",
                         help="Which side of the Double Motor the hammer is plugged into "
                              "(default: left).")
    parser.add_argument("--card-serial", default=None,
                         help="Connection card serial to pin to a specific hub. "
                              "Default: connect to whichever Double Motor advertises first.")
    return parser.parse_args()


def _sides(motor_side):
    """Expands MOTOR_BOTH into separate MOTOR_LEFT/MOTOR_RIGHT calls.

    Sending a single command with the combined MOTOR_BOTH bit mask only
    reliably drives one side in practice (same reason lelib.py's own
    doubleMotor.set_speed() issues two separate per-side calls instead
    of one motor=MOTOR_BOTH call) -- so every action below fans out
    over this instead of passing motor_side straight through."""
    if motor_side == le.MOTOR_BOTH:
        return [le.MOTOR_LEFT, le.MOTOR_RIGHT]
    return [motor_side]


def _mirrored(value, side, sides):
    """Negates value for the right side when driving both sides
    together and MIRROR_RIGHT_SIDE is set -- see that constant."""
    if len(sides) > 1 and MIRROR_RIGHT_SIDE and side == le.MOTOR_RIGHT:
        return -value
    return value


def stop_with_end_state(dm, motor_side, end_state):
    sides = _sides(motor_side)
    for i, side in enumerate(sides):
        blocking = i == len(sides) - 1  # last side blocks so the caller gets a confirmation; others fire immediately
        dm.motor_set_end_state(end_state, motor=side, blocking=False)
        dm.motor_stop(motor=side, blocking=blocking)


def print_telemetry(dm, motor_side):
    if motor_side == le.MOTOR_BOTH:
        sides = [("L", dm.motor[le.MOTOR_LEFT]), ("R", dm.motor[le.MOTOR_RIGHT])]
    else:
        label = "L" if motor_side == le.MOTOR_LEFT else "R"
        sides = [(label, dm.motor[motor_side])]
    for label, m in sides:
        print(f"    [{label}] position={m.position:>6}  absolute={m.absolutePosition:>4}  "
              f"power={m.power:>5}  speed={m.speed:>4}")


def watch(dm, motor_side, seconds):
    if seconds is None:
        print("[watch] streaming telemetry (Ctrl+C to stop)...")
    else:
        print(f"[watch] streaming telemetry for {seconds}s (Ctrl+C to stop early)...")
    start = time.monotonic()
    try:
        while seconds is None or (time.monotonic() - start) < seconds:
            print_telemetry(dm, motor_side)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


def main():
    args = parse_args()
    motor_side = MOTOR_SIDE_BY_NAME[args.motor]

    dm = doubleMotor()
    if args.card_serial is None:
        print("[wacker] Scanning for the closest/first advertising Double Motor "
              "(no connection card needed)...")
    else:
        print(f"[wacker] Connecting to Double Motor with card serial {args.card_serial}...")
    dm.connect(card_serial=args.card_serial)
    print(f"[wacker] Connected. Driving the {args.motor} side.")
    print(__doc__)

    try:
        while True:
            try:
                raw = input("> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()

            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "watch":
                seconds = float(parts[1]) if len(parts) > 1 else None
                watch(dm, motor_side, seconds)
                continue
            elif cmd == "wind":
                degrees = int(parts[1])
                speed = int(parts[2]) if len(parts) > 2 else DEFAULT_WIND_SPEED
                print(f"[wacker] Winding {degrees} deg at {speed}% speed (blocking)...")
                sides = _sides(motor_side)
                for i, side in enumerate(sides):
                    blocking = i == len(sides) - 1
                    dm.motor_run_for_degrees(_mirrored(degrees, side, sides), speed=speed,
                                              motor=side, blocking=blocking)
            elif cmd == "zero":
                sides = _sides(motor_side)
                for i, side in enumerate(sides):
                    blocking = i == len(sides) - 1
                    dm.motor_reset_relative_position(motor=side, blocking=blocking)
                print("[wacker] Position reset to 0.")
            elif cmd == "hold":
                stop_with_end_state(dm, motor_side, le.MOTOR_END_STATE_HOLD)
                print("[wacker] Stopped with end-state HOLD.")
            elif cmd == "brake":
                stop_with_end_state(dm, motor_side, le.MOTOR_END_STATE_BRAKE)
                print("[wacker] Stopped with end-state BRAKE.")
            elif cmd in ("coast", "release"):
                stop_with_end_state(dm, motor_side, le.MOTOR_END_STATE_COAST)
                print("[wacker] Released (end-state COAST) -- spring should snap free now.")
            else:
                try:
                    duty = int(cmd)
                except ValueError:
                    print(f"[wacker] Unrecognized command: {raw!r} (see usage above)")
                    continue
                if not (-100 <= duty <= 100):
                    print("[wacker] Duty cycle must be between -100 and 100.")
                    continue
                sides = _sides(motor_side)
                for i, side in enumerate(sides):
                    blocking = i == len(sides) - 1
                    dm.motor_set_duty_cycle(_mirrored(duty, side, sides), motor=side, blocking=blocking)
                print(f"[wacker] Duty cycle set to {duty}.")

            print_telemetry(dm, motor_side)
    finally:
        print("[wacker] Releasing motor and disconnecting...")
        try:
            stop_with_end_state(dm, motor_side, le.MOTOR_END_STATE_COAST)
        except Exception:
            pass
        dm.disconnect()


if __name__ == "__main__":
    main()
