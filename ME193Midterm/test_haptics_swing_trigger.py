"""
test_haptics_swing_trigger.py

"Unit test #2": builds on test_haptics_trigger.py by replacing the
manual Enter keypress with an automatic trigger -- swing/accelerate the
hammer's own Double Motor hub and it fires trigger_haptic_hit() on its
own, the instant the swing crosses SWING_TRIGGER_THRESHOLD. Still
completely isolated from the game/ball/racquet logic: no camera, no
AprilTag, no second hub -- just this one hub's own accelerometer
driving its own wack.

Swing detection reuses the EXACT same formula real_racquet_input.py
uses for the real racquet (accelerometer magnitude minus the at-rest
gravity baseline, then scaled) -- same sensor hardware, so the same
baseline/scale are a reasonable starting point here too, though this
is a different physical hub and may need its own recalibration (watch
the printed swing_speed while still vs. swinging and adjust
SWING_TRIGGER_THRESHOLD below).

Triggering is edge-based, not time-based: a single swing's
acceleration signal stays above SWING_TRIGGER_THRESHOLD for many
consecutive readings (and can wobble/oscillate while elevated, e.g.
from the hammer's own recoil) -- firing on every reading above
threshold would wack repeatedly for what's physically one swing. So
this only fires on the RISING EDGE (the transition from below
threshold to above it), then "disarms" until the signal drops back
down past SWING_REARM_THRESHOLD -- i.e. the user has to actually stop
and swing again, a second, separate time, before it can fire again.
haptics.trigger_haptic_hit() also enforces its own short cooldown
(HAPTIC_COOLDOWN_S) as a hardware-protection backstop, but the
rising-edge/re-arm logic here is what actually distinguishes "one
swing" from "two swings."

Usage:
    python3 test_haptics_swing_trigger.py
Ctrl+C to stop.
"""

import time

import haptics
from real_racquet_input import ACCEL_MAGNITUDE_BASELINE, ACCEL_TO_SWING_SPEED_SCALE

# CALIBRATE: how much above-baseline swing_speed counts as "swung hard
# enough to wack". Watch the printed swing_speed while gently holding
# the hub still (should stay near 0) vs. actually swinging it (should
# spike clearly above this) and adjust.
SWING_TRIGGER_THRESHOLD = 0.15

# Must drop back below THIS (lower than the trigger threshold, with
# margin) before the next swing is allowed to trigger again. Without
# this gap, sensor noise hovering right at SWING_TRIGGER_THRESHOLD at
# the tail end of a real swing could flicker across it and re-arm/fire
# again within the same physical swing.
SWING_REARM_THRESHOLD = 0.05

POLL_INTERVAL_S = 0.02  # ~50Hz


def main():
    print("Connecting to the hammer's Double Motor hub (same connect_haptics() game.py uses)...")
    haptics.connect_haptics()
    if not haptics._connected:
        print("Not connected -- fix that first (check the warning above) before testing further.")
        return

    print(f"Connected. Swing/accelerate the hub -- swing_speed rising above "
          f"{SWING_TRIGGER_THRESHOLD} fires a wack. It won't fire again until swing_speed "
          f"drops back below {SWING_REARM_THRESHOLD} and rises above the threshold a second, "
          "separate time. Ctrl+C to stop.")

    armed = True  # ready to fire on the next rising edge
    try:
        while True:
            # Re-fetch every iteration: legoeducation REASSIGNS
            # _dm.imu_device to a brand-new object on every BLE
            # notification rather than updating one in place, so a
            # reference captured once before this loop would forever
            # point at the original NaN-filled placeholder from
            # connect time -- exactly what produced a constant 0.000
            # reading before this was fixed.
            imu = haptics._dm.imu_device
            accel_magnitude = (imu.accelerometerX ** 2 + imu.accelerometerY ** 2
                                + imu.accelerometerZ ** 2) ** 0.5
            swing_speed = max(0.0, accel_magnitude - ACCEL_MAGNITUDE_BASELINE) * ACCEL_TO_SWING_SPEED_SCALE

            wacked = False
            if armed and swing_speed >= SWING_TRIGGER_THRESHOLD:
                # trigger_haptic_hit() also enforces its own short
                # cooldown as a hardware-protection backstop, but the
                # rising-edge check above is what actually stops this
                # one sustained swing from firing more than once.
                wacked = haptics.trigger_haptic_hit()
                armed = False
            elif not armed and swing_speed < SWING_REARM_THRESHOLD:
                armed = True

            marker = "  <-- WACK!" if wacked else ("  (disarmed)" if not armed else "")
            print(f"swing_speed={swing_speed:.3f}{marker}")

            time.sleep(POLL_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    finally:
        print("Disconnecting...")
        haptics.disconnect_haptics()


if __name__ == "__main__":
    main()
