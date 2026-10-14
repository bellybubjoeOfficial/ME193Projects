"""
arm_tracker.py

Wraps MediaPipe's HandLandmarker (Tasks API) to track the player's hand/
wrist from the webcam feed -- a second, independent camera-based signal
alongside the AprilTag. Reuses the exact same model/API pattern already
proven in this course's MobileRobots_HW1 project (../main.py,
../src/gestures.py), just reading the wrist position instead of that
project's full gesture vocabulary.

Why track the hand at all when the AprilTag already gives a precise,
calibrated position? A 50mm tag is a small target that's easy to lose
(motion blur mid-swing, going just off-frame -- exactly when it matters
most). A whole hand/arm is a much bigger, more forgiving target for the
camera to keep locked onto, so real_racquet_input.py uses this as a
second vote for both position (preferred fallback during a tag-loss
gap, ahead of blind IMU dead-reckoning) and swing (frame-to-frame wrist
movement, fused with the IMU's accelerometer the same way the
AprilTag's own closing speed already supplements it) -- see that
module's docstring for exactly how they're combined.

No pygame/BLE dependency here -- this is pure camera + MediaPipe, same
"no hardware beyond the camera" scope as apriltag_detector.py.
"""

import os
import urllib.request
from collections import namedtuple

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Standard MediaPipe hand landmark indices (21 points/hand) -- same
# vocabulary as ../src/gestures.py in this course's MobileRobots_HW1.
WRIST = 0
MIDDLE_MCP = 9

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# wrist_x/wrist_y: normalized [0,1] image coordinates (0,0 = top-left,
#   same convention raw MediaPipe/AprilTag pixel positions already use).
# hand_scale: wrist-to-middle-knuckle distance, normalized by frame size
#   -- grows as the hand gets closer to the camera, same role as the
#   AprilTag's own shrinking/growing apparent size for distance. NOT a
#   calibrated real-world distance (unlike the tag's own distance_mm),
#   so real_racquet_input.py only uses this for a relative
#   closing-speed signal, never for distance_mm/in_range directly.
# landmarks: the raw 21-point list, for drawing only (game.py's camera panel).
ArmDetection = namedtuple("ArmDetection", ["wrist_x", "wrist_y", "hand_scale", "landmarks"])


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("[arm-tracker] Downloading hand_landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class ArmTracker:
    def __init__(self):
        _ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb, timestamp_ms):
        """frame_rgb: an RGB (not BGR -- caller must convert) numpy
        array. timestamp_ms must strictly increase between calls
        (MediaPipe's VIDEO running mode requires it) -- pass a
        monotonic millisecond clock, not a frame counter, since a
        dropped camera read would otherwise repeat a timestamp. Returns
        an ArmDetection, or None if no hand was found this frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        landmarks = result.hand_landmarks[0]
        wrist = landmarks[WRIST]
        mcp = landmarks[MIDDLE_MCP]
        hand_scale = ((wrist.x - mcp.x) ** 2 + (wrist.y - mcp.y) ** 2) ** 0.5
        return ArmDetection(wrist_x=wrist.x, wrist_y=wrist.y, hand_scale=hand_scale, landmarks=landmarks)

    def close(self):
        self._landmarker.close()
