"""AprilTag detection via OpenCV's built-in aruco module (no extra deps).

Uses the AprilTag 36h11 dictionary by default - the family printed by most
standard AprilTag generators (e.g. the AprilTag repo's tag36h11 set). If your
tag was printed from a different family, change DEFAULT_DICTIONARY below.
"""

from dataclasses import dataclass

import cv2
import cv2.aruco as aruco
import numpy as np

DEFAULT_DICTIONARY = aruco.DICT_APRILTAG_36h11

# A rotation of pi about the object's X axis: when the tag is held flat,
# facing the camera dead-on, object space (x right, y up, z out of the
# tag toward the camera) and camera space (x right, y down, z into the
# scene) differ by exactly this flip. Removing it from solvePnP's raw
# rotation is what makes estimate_pose() read ~(0, 0, 0) for a tag held
# flat facing the camera, instead of some fixed nonzero offset.
_FLAT_FACING_CAMERA_R = np.diag([1.0, -1.0, -1.0])


@dataclass
class TagDetection:
	tag_id: int
	cx: float
	cy: float
	corners: "cv2.typing.MatLike"  # shape (4, 2), pixel coordinates


class AprilTagDetector:
	def __init__(self, dictionary=DEFAULT_DICTIONARY):
		self._dictionary = aruco.getPredefinedDictionary(dictionary)
		self._params = aruco.DetectorParameters()
		self._detector = aruco.ArucoDetector(self._dictionary, self._params)

	def detect(self, frame_bgr):
		"""Returns the first detected tag as a TagDetection, or None if none found."""
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		corners, ids, _ = self._detector.detectMarkers(gray)

		if ids is None or len(ids) == 0:
			return None

		tag_corners = corners[0][0]  # (4, 2): the 4 tag corners for the first detection
		cx = float(tag_corners[:, 0].mean())
		cy = float(tag_corners[:, 1].mean())
		# ids is (N,) in OpenCV 5.x but (N, 1) in older 4.x releases - ravel
		# handles either shape.
		tag_id = int(ids.ravel()[0])
		return TagDetection(tag_id=tag_id, cx=cx, cy=cy, corners=tag_corners)


def estimate_pose(corners, tag_size_mm, frame_shape):
	"""Estimates the tag's 3D pose from its 4 detected corners: returns
	(pitch_deg, roll_deg, yaw_deg, distance_mm).

	pitch/roll/yaw follow the exact same convention as
	paddle_view._rotation_matrix() (pitch tilts forward/back about X,
	roll tilts left/right about Y, yaw spins in-plane about Z) so they
	can be fed straight into paddle_view.paddle_shape_2d() and
	ball.py's angle-to-lateral-velocity formula with no conversion.

	No camera calibration file is assumed -- this uses a pinhole
	approximation (fx = fy = frame width, principal point at the frame
	center, no lens distortion). That's not a precision measurement,
	but it's enough for a smooth, consistent orientation signal, which
	is all the paddle preview and hit physics need.
	"""
	frame_h, frame_w = frame_shape[:2]
	camera_matrix = np.array([
		[frame_w, 0, frame_w / 2],
		[0, frame_w, frame_h / 2],
		[0, 0, 1],
	], dtype=np.float64)
	dist_coeffs = np.zeros(4)

	s = tag_size_mm / 2
	# Matches detectMarkers()'s corner order: top-left, top-right,
	# bottom-right, bottom-left. Object frame: x right, y up, z out of
	# the tag face (toward the camera when the tag faces it directly).
	object_points = np.array([
		[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0],
	], dtype=np.float64)

	ok, rvec, tvec = cv2.solvePnP(object_points, np.asarray(corners, dtype=np.float64),
								   camera_matrix, dist_coeffs)
	if not ok:
		return None

	rotation, _ = cv2.Rodrigues(rvec)
	relative = _FLAT_FACING_CAMERA_R @ rotation

	# Inverse of R = Rz(yaw) @ Ry(roll) @ Rx(pitch) -- see
	# paddle_view._rotation_matrix() for the forward direction.
	roll = -np.arcsin(np.clip(relative[2, 0], -1.0, 1.0))
	yaw = np.arctan2(relative[1, 0], relative[0, 0])
	pitch = np.arctan2(relative[2, 1], relative[2, 2])

	distance_mm = float(tvec[2, 0])
	return np.degrees(pitch), np.degrees(roll), np.degrees(yaw), distance_mm
