"""
real_racquet_input.py

Real-hardware RacquetInput. Three sensors, each with ONE job:

  - POSITION (self.x/self.y, the racquet circle on screen) comes ONLY
    from MediaPipe hand/wrist tracking (arm_tracker.py). The AprilTag is
    deliberately NOT used for position any more: a 50mm tag regularly
    drops out (off-frame, motion blur mid-swing), and every time it came
    back the racquet snapped from wherever the fallback (hand/IMU
    dead-reckoning) had drifted to back onto the tag -- a visible
    teleport. A whole hand is a much bigger, steadier target, so it's
    now the single source of truth. The wrist position is lightly
    smoothed (POSITION_SMOOTHING_ALPHA) so tracker jitter and the first
    frame after re-acquiring the hand glide instead of jumping. While
    the hand is lost the racquet simply holds its last position (no
    dead-reckoning guesswork to snap back from) and in_range drops to
    False (grey circle).

  - SWING SPEED (how hard the ball comes back) is a weighted blend of
    two camera signals, POSE_SPEED_WEIGHT / TAG_SPEED_WEIGHT (50/50 by
    default): the wrist's own frame-to-frame velocity (plus how fast the
    hand's on-screen size is growing, a stand-in for closing in toward
    the camera), and the AprilTag's frame-to-frame velocity (image-plane
    movement plus how fast its solvePnP distance is shrinking). The tag
    only ever contributes to this speed -- if it isn't visible on two
    consecutive frames, the pose speed is used at 100% rather than being
    halved by a missing vote.

  - SWING DIRECTION (which way the ball is steered) comes from the
    wrist's velocity direction -- the same signal as position, so the
    shot goes the way you visibly moved your hand. Only if the hand
    isn't trackable that frame does the tag's direction stand in.

swing_dir_x/swing_dir_y/swing_dir_z are that direction scaled to the
blended speed: x is left(-)/right(+), y is down(-)/up(+), z is
away(-)/toward(+) the ball (z is floored at 0 -- a backswing contributes
no forward power). ball.py's track_swing()/try_hit() consume these
unchanged.

The Double Motor hub's IMU is no longer part of hit physics or position
at all. It's only used for the paddle PREVIEW's tilt (pitch/roll/yaw),
which is purely cosmetic: vision orientation from the tag when it's
visible, falling back to the hub's own IMU orientation (offset to match
the last tag reading) when it isn't, so the preview doesn't freeze.
SWAP_IMU_PITCH_ROLL documents this hub's mounting quirk: ground-truth
hand-tilt testing found the raw imu.pitch field (not imu.roll) is what
actually carries left/right tilt.

All angle math goes through _wrap_angle_deg()/_smooth_angle_deg()
rather than plain arithmetic because pitch/roll/yaw wrap at +-180
degrees -- naive math across that seam produces ~360-degree blowups.

Connects to the FIRST advertising Double Motor found -- no connection
card needed (see lelib.doubleMotor.connect()).
"""

import time

import cv2

from apriltag_detector import AprilTagDetector, estimate_pose
from arm_tracker import ArmTracker
from lelib import doubleMotor
from racquet_input import RacquetInput


def _wrap_angle_deg(angle_deg):
    """Wraps an angle to (-180, 180]. Without this, any subtraction or
    addition of two angles that straddle the +-180 seam (e.g. +179 and
    -179, a 2-degree real difference) produces a bogus ~360-degree-off
    result."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _smooth_angle_deg(current, target, alpha):
    """Exponential-moving-average smoothing that blends along the
    shortest angular path instead of a naive linear blend (which swings
    wildly through 0 when current/target straddle the +-180 seam)."""
    delta = _wrap_angle_deg(target - current)
    return _wrap_angle_deg(current + alpha * delta)


def _magnitude(vec):
    return (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5


# --- AprilTag (swing speed + cosmetic paddle tilt only) -----------------------
# The real-world edge length of the printed tag's black square. Measure
# racquet_apriltag.png's printed square with a ruler and correct this if
# you printed it at a different size than 50mm.
APRILTAG_SIZE_MM = 50.0

# How much to smooth the paddle preview's pitch/roll/yaw frame-to-frame
# (0 = no smoothing/raw, 1 = frozen forever). Cosmetic only.
ANGLE_SMOOTHING_ALPHA = 0.3

# See module docstring. Only affects the paddle preview's IMU fallback.
SWAP_IMU_PITCH_ROLL = True

# --- Position: hand/wrist only (see module docstring) -------------------------
# How strongly each new wrist reading pulls the on-screen racquet
# toward it per frame (1 = raw/no smoothing, lower = smoother but
# laggier). 0.5 settles in ~4 frames at 60fps -- enough to hide tracker
# jitter and the re-acquire jump without the racquet feeling sluggish.
POSITION_SMOOTHING_ALPHA = 0.5

# --- Swing speed: pose vs. AprilTag blend (see module docstring) --------------
# Must sum to 1.0. Used only when BOTH signals are available on a frame;
# otherwise whichever one is available is used at full weight.
POSE_SPEED_WEIGHT = 0.5
TAG_SPEED_WEIGHT = 0.5

# CALIBRATE THIS: wrist (and tag-center) position is normalized [0,1]
# image coordinates, so a frame-to-frame delta is naturally small -- run
# calibrate_real_input.py, swing at a speed that should definitely
# register as a solid hit, and set this so swing_speed lands in a
# reasonable range (compare against ball.OUTGOING_SPEED_SCALE). Shared
# by the hand and the tag so both measure image-plane speed on the same
# scale and the 50/50 blend is apples-to-apples.
ARM_VELOCITY_TO_SWING_SCALE = 6.0

# CALIBRATE THIS the same way, for the hand-size-growth-rate (closing
# speed) term instead of lateral velocity.
ARM_GROWTH_TO_SWING_SCALE = 6.0

# CALIBRATE THIS: the tag's own closing speed (distance_mm shrinking,
# in mm/second) -> the same swing units as the terms above.
CLOSING_SPEED_TO_SWING_SPEED_SCALE = 1.0 / 3000.0

# Below this, a speed vector is treated as "no direction" (just tracker
# jitter) when choosing whose direction steers the shot.
MIN_DIRECTION_SPEED = 1e-6


class RealRacquetInput(RacquetInput):
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.dm = None
        self.cap = None
        self.detector = AprilTagDetector()
        self.arm_tracker = None

        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.swing_speed = 0.0     # blended pose/tag speed -- magnitude of swing_dir_x/y/z below
        self.swing_dir_x = 0.0     # left(-)/right(+)
        self.swing_dir_y = 0.0     # down(-)/up(+)
        self.swing_dir_z = 0.0     # away(-)/toward(+) the ball, floored at 0
        self.x = 0.0
        self.y = 0.0
        self.in_range = False

        # Exposed for game.py's camera panel / hit-check log and
        # calibrate_real_input.py, not part of the RacquetInput interface.
        self.last_frame_bgr = None
        self.last_detection = None
        self.last_arm_detection = None
        self.hand_seen = False
        self.distance_mm = None
        self.pose_swing_speed = 0.0
        self.tag_swing_speed = 0.0

        # Previous-frame state for velocity estimates -- only valid when
        # the SAME signal was also seen on the previous frame, so a
        # re-acquire after a gap never turns a multi-frame jump into one
        # huge single-frame "swing".
        self._prev_wrist_x = None
        self._prev_wrist_y = None
        self._prev_hand_scale = None
        self._prev_tag_cx = None
        self._prev_tag_cy = None
        self._prev_distance_mm = None
        self._has_position = False

        # Paddle-preview IMU fallback offsets (cosmetic only).
        self._angle_offset_pitch = 0.0
        self._angle_offset_roll = 0.0
        self._angle_offset_yaw = 0.0

    def connect(self):
        self.dm = doubleMotor()
        print("[racquet] Scanning for the closest/first advertising Double Motor "
              "(no connection card needed)...")
        attempt = 0
        while True:
            attempt += 1
            try:
                self.dm.connect()  # no card filter -- first one found wins
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[racquet] Still not connected (attempt {attempt}): {exc} -- retrying...")
        print("[racquet] Connected to Double Motor.")

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Try a different --camera index if this machine has more than one."
            )
        print("[racquet] Camera opened.")

        print("[racquet] Loading hand/arm tracker...")
        self.arm_tracker = ArmTracker()
        print("[racquet] Arm tracker ready.")

    def update(self, dt):
        imu = self.dm.imu_device
        if SWAP_IMU_PITCH_ROLL:
            imu_pitch_raw, imu_roll_raw = float(imu.roll), float(imu.pitch)
        else:
            imu_pitch_raw, imu_roll_raw = float(imu.pitch), float(imu.roll)

        detection = None
        pose = None
        arm_detection = None
        frame_w = frame_h = None
        ok, frame = self.cap.read()
        if ok:
            self.last_frame_bgr = frame
            frame_h, frame_w = frame.shape[:2]
            detection = self.detector.detect(frame)
            if detection is not None:
                pose = estimate_pose(detection.corners, APRILTAG_SIZE_MM, frame.shape)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            arm_detection = self.arm_tracker.detect(frame_rgb, int(time.monotonic() * 1000))
        self.last_detection = detection
        self.last_arm_detection = arm_detection
        self.hand_seen = arm_detection is not None

        # --- Position: hand only ---------------------------------------
        if arm_detection is not None:
            # The camera faces the player, so image left/right is the
            # player's right/left -- negate x so moving your hand to your
            # own right moves the racquet right on screen. Image y+ is
            # down, our y+ is up, so that's negated too.
            target_x = -(arm_detection.wrist_x - 0.5) * 2
            target_y = (arm_detection.wrist_y - 0.5) * -2
            if self._has_position:
                self.x += POSITION_SMOOTHING_ALPHA * (target_x - self.x)
                self.y += POSITION_SMOOTHING_ALPHA * (target_y - self.y)
            else:
                self.x, self.y = target_x, target_y
                self._has_position = True
        self.in_range = arm_detection is not None

        # --- Swing: pose velocity -----------------------------------------
        pose_vec = None
        if arm_detection is not None:
            if self._prev_wrist_x is not None and dt > 0:
                dx = arm_detection.wrist_x - self._prev_wrist_x
                dy = arm_detection.wrist_y - self._prev_wrist_y
                dscale = arm_detection.hand_scale - self._prev_hand_scale
                pose_vec = (-(dx / dt) * ARM_VELOCITY_TO_SWING_SCALE,
                            -(dy / dt) * ARM_VELOCITY_TO_SWING_SCALE,
                            max(0.0, dscale / dt) * ARM_GROWTH_TO_SWING_SCALE)
            self._prev_wrist_x = arm_detection.wrist_x
            self._prev_wrist_y = arm_detection.wrist_y
            self._prev_hand_scale = arm_detection.hand_scale
        else:
            self._prev_wrist_x = self._prev_wrist_y = self._prev_hand_scale = None

        # --- Swing: AprilTag velocity (speed supplement only) -------------
        tag_vec = None
        if pose is not None:
            distance_mm = pose[3]
            tag_cx = detection.cx / frame_w
            tag_cy = detection.cy / frame_h
            if self._prev_tag_cx is not None and dt > 0:
                closing_rate_mm_s = (self._prev_distance_mm - distance_mm) / dt
                tag_vec = (-((tag_cx - self._prev_tag_cx) / dt) * ARM_VELOCITY_TO_SWING_SCALE,
                           -((tag_cy - self._prev_tag_cy) / dt) * ARM_VELOCITY_TO_SWING_SCALE,
                           max(0.0, closing_rate_mm_s) * CLOSING_SPEED_TO_SWING_SPEED_SCALE)
            self._prev_tag_cx, self._prev_tag_cy = tag_cx, tag_cy
            self._prev_distance_mm = distance_mm
            self.distance_mm = distance_mm
        else:
            self._prev_tag_cx = self._prev_tag_cy = self._prev_distance_mm = None

        # --- Swing: blend --------------------------------------------------
        self.pose_swing_speed = _magnitude(pose_vec) if pose_vec is not None else 0.0
        self.tag_swing_speed = _magnitude(tag_vec) if tag_vec is not None else 0.0
        if pose_vec is not None and tag_vec is not None:
            speed = POSE_SPEED_WEIGHT * self.pose_swing_speed + TAG_SPEED_WEIGHT * self.tag_swing_speed
        elif pose_vec is not None:
            speed = self.pose_swing_speed
        elif tag_vec is not None:
            speed = self.tag_swing_speed
        else:
            speed = 0.0

        if self.pose_swing_speed > MIN_DIRECTION_SPEED:
            direction_vec, direction_mag = pose_vec, self.pose_swing_speed
        elif self.tag_swing_speed > MIN_DIRECTION_SPEED:
            direction_vec, direction_mag = tag_vec, self.tag_swing_speed
        else:
            direction_vec, direction_mag = None, 0.0

        if direction_vec is not None:
            self.swing_dir_x = direction_vec[0] / direction_mag * speed
            self.swing_dir_y = direction_vec[1] / direction_mag * speed
            self.swing_dir_z = direction_vec[2] / direction_mag * speed
        else:
            self.swing_dir_x = self.swing_dir_y = self.swing_dir_z = 0.0
        self.swing_speed = speed

        # --- Paddle preview tilt (cosmetic) --------------------------------
        if pose is not None:
            pitch, roll, yaw, _ = pose
            self._angle_offset_pitch = _wrap_angle_deg(pitch - imu_pitch_raw)
            self._angle_offset_roll = _wrap_angle_deg(roll - imu_roll_raw)
            self._angle_offset_yaw = _wrap_angle_deg(yaw - float(imu.yaw))
        else:
            pitch = _wrap_angle_deg(imu_pitch_raw + self._angle_offset_pitch)
            roll = _wrap_angle_deg(imu_roll_raw + self._angle_offset_roll)
            yaw = _wrap_angle_deg(float(imu.yaw) + self._angle_offset_yaw)
        self.pitch = _smooth_angle_deg(self.pitch, pitch, ANGLE_SMOOTHING_ALPHA)
        self.roll = _smooth_angle_deg(self.roll, roll, ANGLE_SMOOTHING_ALPHA)
        self.yaw = _smooth_angle_deg(self.yaw, yaw, ANGLE_SMOOTHING_ALPHA)

    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
        if self.arm_tracker is not None:
            self.arm_tracker.close()
        if self.dm is not None:
            try:
                self.dm.disconnect()
            except Exception as exc:
                print(f"[racquet] Error while disconnecting: {exc}")
