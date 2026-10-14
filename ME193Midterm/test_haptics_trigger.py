"""
test_haptics_trigger.py

Minimal standalone test for haptics.py, completely isolated from the
game/ball/racquet logic: connects using the EXACT same connect_haptics()
game.py calls, then lets you fire the wack pulse on demand by pressing
Enter, using the EXACT same trigger_haptic_hit() game.py calls.

This isolates whether the hardware path itself works (does the motor
actually spin when told to?) from whether the GAME ever reaches that
call (hit detection, game loop wiring, etc.). If the motor still
doesn't spin here, the problem is in haptics.py or the hardware itself,
not in game.py's hit-detection logic -- since this script never touches
ball.py/real_racquet_input.py/game.py at all.

Usage:
    python3 test_haptics_trigger.py
Press Enter to fire a wack. Type q + Enter to quit.
"""

import legoeducation as le

import haptics


def print_telemetry():
    """Reaches into haptics.py's internals (_dm, _connected, _sides()) on
    purpose -- this is a debug tool for this project, not a public API,
    so it's fine to read the same state haptics.py itself tracks."""
    if not haptics._connected:
        print("    (not connected -- no telemetry)")
        return
    for side in haptics._sides():
        m = haptics._dm.motor[side]
        label = "L" if side == le.MOTOR_LEFT else "R"
        print(f"    [{label}] position={m.position:>6}  absolute={m.absolutePosition:>4}  "
              f"power={m.power:>5}  speed={m.speed:>4}")


def main():
    print(f"Connecting to the hammer's Double Motor hub (HAMMER_MOTOR_SIDE="
          f"{'BOTH' if haptics.HAMMER_MOTOR_SIDE == le.MOTOR_BOTH else haptics.HAMMER_MOTOR_SIDE}, "
          f"same connect_haptics() game.py calls)...")
    haptics.connect_haptics()

    if not haptics._connected:
        print("Not connected -- fix that first (check the warning above) before testing further.")
        return

    print("Connected.")
    print_telemetry()
    print()
    print("Press Enter to fire a wack (the exact trigger_haptic_hit() game.py calls on every hit).")
    print("Type q + Enter to quit.")

    try:
        while True:
            raw = input("> ").strip().lower()
            if raw == "q":
                break
            print("Before:")
            print_telemetry()
            haptics.trigger_haptic_hit()
            print("After (telemetry may lag the physical motion by a frame or two):")
            print_telemetry()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        print("Disconnecting...")
        haptics.disconnect_haptics()


if __name__ == "__main__":
    main()
