"""
racquet_input.py

Defines the interface every racquet input source implements (pitch,
roll, yaw, swing_speed, swing_dir_x/y/z, lateral x/y, in_range), plus
SimulatedRacquetInput -- a mouse/keyboard stand-in for the real
camera+AprilTag (distance/position) + BLE IMU (swing) sensing.
Swapping to the real sensors later only requires implementing this
same interface in a new class (e.g. RealRacquetInput) -- game.py and
ball.py never need to change.

Hit physics is driven entirely by swing_dir_x/y/z (see ball.py) --
pitch/roll/yaw are cosmetic only (the on-screen paddle preview's tilt),
same as in RealRacquetInput.

Simulated controls:
    Mouse X position      -> roll (cosmetic paddle tilt left/right only)
    Mouse Y position       -> depth/distance: bottom of the window = in
                              range to hit, top = out of range
    Mouse X *velocity*     -> swing_dir_x (steer left/right)
    Mouse Y *velocity*,
      downward only        -> swing_dir_z (forward power) -- moving the
                              mouse down fast is the stand-in for
                              "swinging the racquet forward fast," which
                              also happens to be exactly what closes the
                              distance gap in real ping pong too
    Q / E (held)           -> swing_dir_y (steer down/up) and, same as
                              before, cosmetic pitch tilt; springs back
                              toward 0 when released
"""

import pygame

PITCH_RATE_DEG_PER_S = 90.0     # how fast Q/E tilts pitch while held
PITCH_SPRING_BACK = 3.0         # how fast pitch relaxes toward 0 when not held
ROLL_MAX_DEG = 45.0
PITCH_MAX_DEG = 45.0
IN_RANGE_Y_FRACTION = 0.6       # mouse below this fraction of window height = in range

SWING_DIR_Y_RATE = 2.0          # how fast Q/E ramps swing_dir_y (up/down steer) while held
SWING_DIR_Y_SPRING_BACK = 4.0   # how fast swing_dir_y relaxes toward 0 when not held


class RacquetInput:
    """Interface every input source implements."""

    def connect(self):
        """Any one-time setup (hardware connections, opening a camera,
        etc.). No-op by default -- SimulatedRacquetInput needs nothing."""
        pass

    def update(self, dt):
        """Refreshes pitch/roll/yaw/swing_speed/swing_dir_x/y/z/x/y/
        in_range for this frame."""
        raise NotImplementedError

    def disconnect(self):
        """Any one-time teardown. No-op by default."""
        pass


class SimulatedRacquetInput(RacquetInput):
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.swing_speed = 0.0
        self.swing_dir_x = 0.0
        self.swing_dir_y = 0.0
        self.swing_dir_z = 0.0
        self.x = 0.0
        self.y = 0.0
        self.in_range = False
        self._prev_mouse_x = None
        self._prev_mouse_y = None

    def update(self, dt):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()

        self.roll = ((mouse_x / self.screen_width) - 0.5) * 2 * ROLL_MAX_DEG

        if keys[pygame.K_q]:
            self.pitch = max(-PITCH_MAX_DEG, self.pitch - PITCH_RATE_DEG_PER_S * dt)
            self.swing_dir_y = max(-1.0, self.swing_dir_y - SWING_DIR_Y_RATE * dt)
        elif keys[pygame.K_e]:
            self.pitch = min(PITCH_MAX_DEG, self.pitch + PITCH_RATE_DEG_PER_S * dt)
            self.swing_dir_y = min(1.0, self.swing_dir_y + SWING_DIR_Y_RATE * dt)
        else:
            self.pitch *= max(0.0, 1.0 - PITCH_SPRING_BACK * dt)
            self.swing_dir_y *= max(0.0, 1.0 - SWING_DIR_Y_SPRING_BACK * dt)

        self.in_range = mouse_y > self.screen_height * IN_RANGE_Y_FRACTION

        if self._prev_mouse_x is not None and dt > 0:
            dx = mouse_x - self._prev_mouse_x
            self.swing_dir_x = dx / self.screen_width / dt
        self._prev_mouse_x = mouse_x

        if self._prev_mouse_y is not None and dt > 0:
            dy = mouse_y - self._prev_mouse_y
            self.swing_dir_z = max(0.0, dy) / self.screen_height / dt
        self._prev_mouse_y = mouse_y
        self.swing_speed = (self.swing_dir_x ** 2 + self.swing_dir_y ** 2 + self.swing_dir_z ** 2) ** 0.5

        self.x = ((mouse_x / self.screen_width) - 0.5) * 2
        self.y = ((mouse_y / self.screen_height) - 0.5) * -2  # invert so up = +y
