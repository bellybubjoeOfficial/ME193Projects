"""
paddle_view.py

Pure math for the small 3D-looking paddle preview (given its live
pitch/roll/yaw) -- a simple rotate-and-project approach (no 3D engine needed for a flat paddle silhouette). No pygame,
fully testable in isolation.

The paddle is modeled as two flat pieces in local XY (z=0) space, both
rotated/projected together so they stay rigidly attached as one object:
a circular blade and a rectangular handle below it.
"""

import numpy as np


BLADE_RADIUS = 0.55
BLADE_SEGMENTS = 24
BLADE_CENTER_Y = 0.15     # blade sits above the origin, handle below it

HANDLE_WIDTH = 0.18
HANDLE_LENGTH = 0.45
HANDLE_TOP_Y = BLADE_CENTER_Y - BLADE_RADIUS + 0.05   # slight overlap into the blade


def _rotation_matrix(pitch_deg, roll_deg, yaw_deg):
    """pitch tilts forward/back (rotate about the horizontal X axis), roll
    tilts left/right (rotate about the vertical Y axis -- this is what
    makes the paddle face visibly foreshorten side-to-side, which is what
    "roll" means for a held paddle; it's yaw, not roll, that spins the
    paddle in-plane like a clock hand around its own forward axis)."""
    p, r, y = np.radians([pitch_deg, roll_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _blade_points_3d():
    angles = np.linspace(0, 2 * np.pi, BLADE_SEGMENTS, endpoint=False)
    xs = BLADE_RADIUS * np.cos(angles)
    ys = BLADE_RADIUS * np.sin(angles) + BLADE_CENTER_Y
    return np.column_stack([xs, ys, np.zeros_like(xs)])


def _handle_points_3d():
    hw = HANDLE_WIDTH / 2
    y_bottom = HANDLE_TOP_Y - HANDLE_LENGTH
    return np.array([
        [-hw, HANDLE_TOP_Y, 0.0], [hw, HANDLE_TOP_Y, 0.0],
        [hw, y_bottom, 0.0], [-hw, y_bottom, 0.0],
    ])


def _project(points_3d, rotation, focal_length):
    rotated = points_3d @ rotation.T
    rotated[:, 2] += focal_length  # push the paddle in front of the camera
    return rotated[:, :2] / rotated[:, 2:3] * focal_length


def paddle_shape_2d(pitch_deg, roll_deg, yaw_deg=0.0, focal_length=3.0):
    """Returns {"blade": Nx2 points, "handle": 4x2 points}, both rotated by
    the given angles and projected to 2D via a simple perspective
    projection. Caller scales/translates to screen pixels and draws each
    as its own filled polygon."""
    rotation = _rotation_matrix(pitch_deg, roll_deg, yaw_deg)
    return {
        "blade": _project(_blade_points_3d(), rotation, focal_length),
        "handle": _project(_handle_points_3d(), rotation, focal_length),
    }

