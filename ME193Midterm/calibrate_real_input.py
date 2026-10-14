"""
calibrate_real_input.py

Connects to the Double Motor + camera (same as real_racquet_input.py) and
prints live readings with no game logic -- use this to tune
IN_RANGE_DISTANCE_MM, ACCEL_MAGNITUDE_BASELINE, ACCEL_TO_SWING_SPEED_SCALE,
CLOSING_SPEED_TO_SWING_SPEED_SCALE, ACCEL_TO_POSITION_SCALE,
ACCEL_TO_DISTANCE_SCALE, ACCEL_X_BASELINE/ACCEL_Y_BASELINE/ACCEL_Z_BASELINE,
ARM_VELOCITY_TO_SWING_SCALE/ARM_GROWTH_TO_SWING_SCALE, and
SWAP_IMU_PITCH_ROLL at the top of real_racquet_input.py before playing:

  - Watch distance_mm while holding the racquet at the depth that
    should count as "in range to hit," and set IN_RANGE_DISTANCE_MM
    just above that reading.
  - Watch accel_magnitude while holding the racquet completely still --
    it reads gravity, not 0 (expect well above zero, consistently).
    Set ACCEL_MAGNITUDE_BASELINE to that resting value.
  - Watch accel_magnitude while holding the racquet still (should now
    read near the baseline above) and while swinging at a speed that
    should definitely register as a solid hit (should be clearly
    larger). Pick ACCEL_TO_SWING_SPEED_SCALE so that
    (still-reading - ACCEL_MAGNITUDE_BASELINE) * scale stays well below
    a real swing's (reading - baseline) * scale.
  - Cover the tag, tilt the racquet left/right, and watch whether pitch
    or roll actually changes -- on this hub's known mounting, pitch
    carries left-right tilt, not roll (SWAP_IMU_PITCH_ROLL handles
    this), but confirm it on any newly-mounted or replaced hub.
  - Watch closing_rate_mm_s while holding the racquet still (should be
    near 0) and while moving it toward the camera at a speed that
    should definitely register as a fast hit. Pick
    CLOSING_SPEED_TO_SWING_SPEED_SCALE so that reading lands in the
    same range as the accelerometer term above -- swing_speed is
    whichever of the two terms is larger each frame.
  - pitch/roll/yaw here are the smoothed, AprilTag-vision-derived
    angles -- cosmetic only now (the paddle preview's tilt), not used
    by hit physics -- tilt the racquet and confirm they track smoothly
    with no jumps.
  - Watch swing_dir_x/y/z while holding the racquet completely still
    (facing the camera, normal grip) -- each should read ~0. If one
    doesn't, set that axis's ACCEL_*_BASELINE to the raw accelerometer
    value it reads while still. Then swing left/right, up/down, and
    straight forward one at a time and confirm each produces a clear
    spike on ONLY its own axis (swing_dir_x for left/right, swing_dir_y
    for up/down, swing_dir_z for forward) with the others near 0; flip
    a constant's sign or swap which axis feeds which if a swing shows
    up on the wrong one or the wrong sign.
  - To tune ACCEL_TO_POSITION_SCALE/ACCEL_TO_DISTANCE_SCALE: cover the
    tag AND keep your hand out of frame for under a second while moving
    the racquet side to side and toward/away from the camera, watching
    x/y/distance_mm (tag_seen=False, hand_seen=False) keep tracking
    plausibly through the gap instead of freezing. If the dead-reckoned
    motion goes the wrong direction, flip that constant's sign; if it
    barely moves, increase its magnitude.
  - hand_seen shows whether arm_tracker.py currently sees your hand --
    hold it up in frame (same hand that holds the racquet) and confirm
    it stays True while you move it around the frame. Cover the tag but
    keep your hand visible, and confirm x/y now track your hand's
    actual on-screen position directly (not a drifting dead-reckoned
    guess) -- this is the preferred tag-loss fallback; IMU dead-reckoning
    only kicks in once hand_seen also goes False.
  - To tune ARM_VELOCITY_TO_SWING_SCALE/ARM_GROWTH_TO_SWING_SCALE: watch
    swing_dir_x/y/z while holding your hand still in frame (should stay
    near 0, same as the IMU-based check above), then swing your arm
    left/right, up/down, and straight toward the camera one at a time --
    each should spike on only its own axis, in the same range as the
    IMU-driven spikes already tuned above (both are fused together via
    "whichever reads stronger wins," so wildly mismatched scales would
    let one signal always dominate the other).

Usage:
    python calibrate_real_input.py
    python calibrate_real_input.py --camera 1
Ctrl+C to stop.
"""

import argparse
import time

from real_racquet_input import RealRacquetInput


def parse_args():
    parser = argparse.ArgumentParser(description="Print live racquet sensor readings for calibration")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    return parser.parse_args()


def main():
    args = parse_args()
    racquet = RealRacquetInput(camera_index=args.camera)
    racquet.connect()

    print("Printing live readings. Ctrl+C to stop.")
    last_time = time.monotonic()
    prev_distance_mm = None  # only valid when the previous iteration also saw the tag
    try:
        while True:
            now = time.monotonic()
            dt = now - last_time
            last_time = now

            racquet.update(dt=dt)
            tag_seen_now = racquet.last_detection is not None

            imu = racquet.dm.imu_device
            accel_magnitude = (imu.accelerometerX ** 2 + imu.accelerometerY ** 2
                                + imu.accelerometerZ ** 2) ** 0.5
            if tag_seen_now and prev_distance_mm is not None and dt > 0:
                closing_rate_mm_s = (prev_distance_mm - racquet.distance_mm) / dt
            else:
                closing_rate_mm_s = 0.0
            prev_distance_mm = racquet.distance_mm if tag_seen_now else None

            distance_str = f"{racquet.distance_mm:7.1f}" if racquet.distance_mm is not None else "   n/a "
            print(f"tag_seen={tag_seen_now!s:5}  hand_seen={racquet.hand_seen!s:5}  "
                  f"pitch={racquet.pitch:+6.1f}  roll={racquet.roll:+6.1f}  yaw={racquet.yaw:+6.1f}   "
                  f"accel_magnitude={accel_magnitude:8.1f}  closing_rate_mm_s={closing_rate_mm_s:+9.1f}  "
                  f"swing_speed={racquet.swing_speed:.3f}  "
                  f"swing_dir=({racquet.swing_dir_x:+.2f},{racquet.swing_dir_y:+.2f},{racquet.swing_dir_z:+.2f})   "
                  f"distance_mm={distance_str}  in_range={racquet.in_range}   "
                  f"x={racquet.x:+.2f} y={racquet.y:+.2f}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        racquet.disconnect()


if __name__ == "__main__":
    main()
