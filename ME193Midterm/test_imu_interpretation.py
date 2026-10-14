"""
test_imu_interpretation.py

A readable, standalone diagnostic for "which way does the game think
the ball goes, given this swing data" -- no hardware needed. Prints
human-readable verdicts (BALL GOES RIGHT / LEFT / UP / DOWN / FORWARD
fast-or-slow) for a set of named physical scenarios, covering the whole
chain in two independently-checkable stages:

  1. ball.py's physics alone (swing_x/y/z -> outgoing vx/vy/vz) -- pure
     math, fully verifiable without any hardware. Physics is
     gesture-based: exactly ONE axis of the racquet's swing -- whichever
     of back-and-forth (power), left/right, or up/down changed the most
     -- decides the outgoing lateral direction; the other two
     contribute nothing. Hit registration itself is purely
     position-gated (no minimum swing required at all -- the racquet's
     circle behaves like a solid boundary that always ricochets the
     ball on contact); paddle orientation (pitch/roll) plays no part in
     this at all any more -- see ball.py's module docstring.
  2. RealRacquetInput's always-on accelerometer -> swing_dir_x/y/z
     path (synthetic IMU readings, no hardware needed) -- confirming
     the swing signal keeps flowing identically whether or not the
     AprilTag is visible that frame, and that the tag's own closing
     speed only ever supplements the forward/power axis (swing_dir_z),
     never steering. The ONE thing this can't verify without the
     physical hub is the exact sign/axis mapping of its raw
     accelerometer fields to the racquet's real-world left/right/up/
     down/forward axes (same caveat as ACCEL_TO_POSITION_SCALE) --
     confirm that on hardware with calibrate_real_input.py.

Run: python3 test_imu_interpretation.py
"""

import numpy as np

from ball import MIN_OUTGOING_SPEED, OUTGOING_SPEED_SCALE, Ball

LATERAL_DEADZONE = 0.02  # vx/vy below this magnitude is reported as "no sideways drift"


def describe_vx(vx):
    if vx > LATERAL_DEADZONE:
        return f"vx={vx:+.3f} -> BALL GOES RIGHT"
    if vx < -LATERAL_DEADZONE:
        return f"vx={vx:+.3f} -> BALL GOES LEFT"
    return f"vx={vx:+.3f} -> no left/right drift"


def describe_vy(vy):
    if vy > LATERAL_DEADZONE:
        return f"vy={vy:+.3f} -> BALL GOES UP"
    if vy < -LATERAL_DEADZONE:
        return f"vy={vy:+.3f} -> BALL GOES DOWN"
    return f"vy={vy:+.3f} -> no up/down drift"


def describe_vz(vz):
    speed = "FAST" if vz > MIN_OUTGOING_SPEED * 1.5 else "slow"
    return f"vz={vz:+.3f} -> BALL GOES FORWARD ({speed})"


def hit_with(swing_x, swing_y, swing_z, frames=10):
    """Runs the actual production code path (track_swing() for `frames`
    frames, then a real Ball's try_hit()) with the racquet positioned
    right on top of the ball so position/depth always qualify -- hit
    registration is purely position-gated, so this always succeeds;
    the swing vector only affects the resulting outgoing vx/vy/vz.
    Returns None if try_hit() ever does reject (kept for symmetry/
    robustness), else (vx, vy, vz)."""
    ball = Ball()
    ball.x, ball.y, ball.z = 0.0, 0.0, 0.0
    ball.state = "approaching"
    dt = 1 / 60
    for _ in range(frames):
        ball.track_swing(swing_x, swing_y, swing_z, dt)
    hit = ball.try_hit(racquet_x=0.0, racquet_y=0.0)
    if not hit:
        return None
    return ball.vx, ball.vy, ball.vz


def section(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def run_ball_physics_section():
    section("1) ball.py physics alone (swing_x/y/z -> outgoing velocity)")
    print(f"(OUTGOING_SPEED_SCALE={OUTGOING_SPEED_SCALE}, MIN_OUTGOING_SPEED={MIN_OUTGOING_SPEED})")
    scenarios = [
        ("Pure RIGHTWARD swing -> steers right, still fast forward (power is magnitude-based)",
         2.0, 0.0, 0.0),
        ("Pure LEFTWARD swing -> steers left", -2.0, 0.0, 0.0),
        ("Pure UPWARD swing -> steers up", 0.0, 2.0, 0.0),
        ("Pure DOWNWARD swing -> steers down", 0.0, -2.0, 0.0),
        ("Pure forward swing (straight toward the ball), no lateral component -> straight, fast",
         0.0, 0.0, 2.0),
        ("Motionless touch (no swing at all) -> still bounces (floor speed), purely position-gated",
         0.0, 0.0, 0.0),
        ("STRONG forward swing with a little INCIDENTAL sideways noise -> forward dominates, steering suppressed",
         0.3, 0.0, 3.0),
        ("Weak forward swing with a BIG deliberate sideways swing -> sideways dominates, steers the ball",
         2.5, 0.0, 0.3),
    ]
    for label, sx, sy, sz in scenarios:
        result = hit_with(sx, sy, sz)
        print(f"\n{label}")
        print(f"  swing=(x={sx:+.2f}, y={sy:+.2f}, z={sz:+.2f})")
        if result is None:
            print("  NO HIT (unexpected -- hit registration is purely position-gated now)")
            continue
        vx, vy, vz = result
        print(f"  {describe_vx(vx)}")
        print(f"  {describe_vy(vy)}")
        print(f"  {describe_vz(vz)}")

    print("\n-- Forgiveness check: a swing shortly before contact still counts --")
    print("A strong rightward swing happens, then the racquet goes quiet for 5 frames before")
    print("the ball actually arrives (the exact scenario the user asked to make forgiving).")
    ball = Ball()
    ball.x, ball.y, ball.z = 0.0, 0.0, 0.0
    ball.state = "approaching"
    dt = 1 / 60
    ball.track_swing(2.0, 0.0, 0.0, dt)
    for _ in range(5):
        ball.track_swing(0.0, 0.0, 0.0, dt)  # quiet right at the contact frame
    hit = ball.try_hit(racquet_x=0.0, racquet_y=0.0)
    print(f"  hit={hit} vx={ball.vx:+.3f} vy={ball.vy:+.3f} vz={ball.vz:+.3f}")
    assert hit and ball.vx > 0.1, "a swing shortly before contact should still steer the ball"
    print("  OK: still steers right, at somewhat reduced strength from the decay")


def run_real_racquet_input_section():
    section("2) RealRacquetInput: always-on IMU swing (synthetic accelerometer, no hardware needed)")
    print("Confirms swing_dir_x/y/z keep flowing identically whether or not the AprilTag is")
    print("visible that frame, and that the tag's closing speed only ever supplements the")
    print("forward/power axis (swing_dir_z) -- never steering. Axis SIGN/mapping itself (which")
    print("raw accelerometer field is really left/right vs. up/down on this hub's mounting)")
    print("can't be verified without the physical hub -- confirm with calibrate_real_input.py.")

    from real_racquet_input import ACCEL_TO_SWING_SPEED_SCALE, ACCEL_X_BASELINE, ACCEL_Y_BASELINE, \
        ACCEL_Z_BASELINE, RealRacquetInput

    class FakeImu:
        def __init__(self):
            self.pitch = 0.0
            self.roll = 0.0
            self.yaw = 0.0
            self.accelerometerX = ACCEL_X_BASELINE
            self.accelerometerY = ACCEL_Y_BASELINE
            self.accelerometerZ = ACCEL_Z_BASELINE

    class FakeDM:
        def __init__(self):
            self.imu_device = FakeImu()

    class FakeCap:
        def __init__(self):
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def read(self):
            return True, self.frame

    class FakeDetector:
        def __init__(self):
            self.detection = None

        def detect(self, frame):
            return self.detection

    from apriltag_detector import TagDetection

    def flat_detection():
        return TagDetection(tag_id=0, cx=320, cy=240,
                             corners=np.array([[280, 200], [360, 200], [360, 280], [280, 280]], dtype=np.float32))

    def make_input():
        r = RealRacquetInput()
        r.dm = FakeDM()
        r.cap = FakeCap()
        r.detector = FakeDetector()
        return r

    # -- Swing reads the same whether the tag is visible or lost. --
    delta = 400.0  # raw accelerometer units away from baseline, on X (left/right)
    expected = delta * ACCEL_TO_SWING_SPEED_SCALE

    r = make_input()
    r.detector.detection = flat_detection()
    r.dm.imu_device.accelerometerX = ACCEL_X_BASELINE + delta
    r.update(dt=1 / 30)
    print(f"\nTag VISIBLE, accelerometerX +{delta:.0f} above baseline:")
    print(f"  swing_dir_x={r.swing_dir_x:+.3f} (expected {expected:+.3f})")
    assert abs(r.swing_dir_x - expected) < 1e-6

    r2 = make_input()
    r2.detector.detection = None  # tag lost this whole time
    r2.dm.imu_device.accelerometerX = ACCEL_X_BASELINE + delta
    r2.update(dt=1 / 30)
    print("Tag LOST the entire time, same accelerometerX reading:")
    print(f"  swing_dir_x={r2.swing_dir_x:+.3f} (expected {expected:+.3f}, same as tag-visible case)")
    assert abs(r2.swing_dir_x - expected) < 1e-6
    print("  OK: swing tracking is unaffected by tag visibility, as intended")

    # -- AprilTag closing speed supplements swing_dir_z specifically, never x/y. --
    r3 = make_input()
    r3.dm.imu_device.accelerometerX = ACCEL_X_BASELINE + delta  # steering signal present
    # Frame 1: establish a distance baseline.
    r3.detector.detection = flat_detection()
    r3.update(dt=1 / 30)
    # Frame 2: tag reads a much closer distance -- fast closing speed, bigger
    # corners (tag appears bigger as it gets closer).
    close_detection = TagDetection(
        tag_id=0, cx=320, cy=240,
        corners=np.array([[200, 120], [440, 120], [440, 360], [200, 360]], dtype=np.float32))
    r3.detector.detection = close_detection
    r3.update(dt=1 / 30)
    print(f"\nFast AprilTag closing speed, weak forward accelerometer reading:")
    print(f"  swing_dir_z={r3.swing_dir_z:+.3f} (boosted above the accelerometer-only reading)")
    print(f"  swing_dir_x={r3.swing_dir_x:+.3f} (unaffected -- closing speed never steers)")
    assert abs(r3.swing_dir_x - expected) < 1e-6
    print("  OK: closing speed only ever supplements the forward/power axis")


if __name__ == "__main__":
    run_ball_physics_section()
    run_real_racquet_input_section()
    print()
