"""
ball.py

Pure ball/game-state logic for virtual ping pong: the ball's approach
trajectory, the hit-detection window, reflection physics (outgoing
speed and direction from the racquet's swing -- a 3-axis acceleration
vector -- at the moment of the hit), and scoring against a target zone.
No pygame, no hardware -- fully testable in isolation (same "pure logic
separated from I/O" philosophy as q_learning.py/whistle_detector.py
elsewhere in this project).

Physics is driven entirely by the racquet's SWING (translational
acceleration): back-and-forth motion (toward/away from the ball) is
power, left-right motion steers left/right, up-down motion steers
up/down -- see track_swing()/try_hit(). Paddle orientation (pitch/roll
from the AprilTag) no longer affects ball physics at all; it's purely
a position/attitude display concern now (see game.py's paddle preview).
Position (is the racquet laterally close enough to the ball at all) is
still AprilTag-driven and only a coarse gate, same as before -- see
would_hit()/try_hit().

Coordinate space (abstract units, not pixels -- game.py maps these to
screen coordinates):
    z: depth. 1.0 = far screen (where the target zone is), 0.0 = at the
       player/racquet.
    x, y: lateral position on the screen plane, roughly [-1, 1].
"""

import random

APPROACH_SPEED = 0.4       # z-units/second the ball closes while approaching
HIT_ZONE_DEPTH = 0.15      # how close to z=0 counts as "at the racquet" (lenient timing window)
HIT_DISTANCE_MAX = 0.55    # max lateral (x/y) distance between ball and racquet to count as a hit (lenient aim)

# game.py's to_screen() renders y-axis lateral movement more compressed
# than x (purely to keep the play area's vertical extent on screen --
# see its own comment) -- a given WORLD-SPACE y distance ends up as only
# this fraction as many on-screen pixels as the same x distance. Every
# "is this close enough to count as touching" check below (the racquet's
# hit circle, the target zone) is drawn as an ISOTROPIC circle/box in
# PIXELS, so a plain Euclidean distance in raw x/y world-units doesn't
# match what's actually drawn: it under-tolerates y-direction touches
# that visually clear the circle/box just fine, because the same raw y
# distance looks much smaller on screen than the same raw x distance
# does. visual_lateral_distance() below corrects for this so "touches
# on-screen" and "counts as a hit/score" always agree. game.py imports
# this SAME constant for to_screen() itself, so the two can never drift
# out of sync with each other again.
LATERAL_Y_SCREEN_SQUISH = 0.3

TARGET_ZONE_HALF_SIZE = 0.3      # the scoring region's half-width/height, same size each shot
TARGET_ZONE_CENTER_RANGE = 0.25  # how far the target zone's center can randomize each shot

# The ball's own radius in these same world-space units, at z=1 (the far
# wall, where it's smallest). Target-zone scoring itself is decided by
# game.py in actual screen pixels (see lands_in_target_zone_px()),
# not by this constant directly -- this exists so that game.py's drawn
# far-ball radius (BALL_MIN_RADIUS) is derived from a documented,
# named world-space value instead of an arbitrary pixel count.
BALL_RADIUS = 0.06

# swing magnitude -> outgoing speed multiplier (any axis). Halved from
# 2.0: medium-speed swings were already launching the ball back almost
# instantly. Only scales how much the swing ADDS -- the near-motionless
# touch case is set by MIN_OUTGOING_SPEED below and is unchanged.
OUTGOING_SPEED_SCALE = 1.0
MIN_OUTGOING_SPEED = 0.1        # floor so even a near-motionless touch still visibly bounces back

# When a shot lands in the target box, the ball bounces off the far wall
# and comes back at this fraction of the speed it arrived with (so a
# hard shot comes back hard -- momentum), becoming the next incoming
# ball of the rally. See bounce_off_wall().
WALL_BOUNCE_SPEED_FRACTION = 0.8

# Floor on how fast a bounced ball comes back toward you: 80% of a
# near-motionless touch (MIN_OUTGOING_SPEED) would take over 10 seconds
# to arrive. Faster shots are unaffected.
MIN_BOUNCE_APPROACH_SPEED = 0.25

# How far (in x/y world units) the steering axis can push the ball's
# landing spot away from where it was hit, for a swing that's purely
# sideways/vertical. Previously the steer velocity equalled the forward
# speed outright, so ANY swing whose strongest axis was sideways
# launched at 45 degrees and landed ~1 unit off -- well outside the
# target box (half-size TARGET_ZONE_HALF_SIZE) every time. Scaled to
# the box size instead, so a deliberate steer moves the shot across the
# box rather than off the screen.
MAX_STEER_OFFSET = 0.45

# How long a momentum spike's "memory" lasts while tracking the peak
# swing vector during the approach (see track_swing()) -- lets a real
# swing that peaks a few frames before/after the exact contact frame
# still count at full strength (both its power AND its direction),
# instead of only whatever instantaneous reading lands on that one
# frame. This is the whole mechanism behind "a swing right before
# contact should count": the peak is held and decays slowly, it isn't
# discarded the instant the hand stops accelerating.
SWING_SPEED_HALF_LIFE = 0.25

# How far ahead of actual contact (in seconds) to fire the haptic wack
# -- compensates for the physical motor pulse's own execution latency,
# since firing exactly AT the moment of visual contact would make the
# physical smack land noticeably late. Purely a haptic-timing concern:
# it does NOT change scoring/bounce physics at all -- those are still
# decided at the real contact moment by try_hit(). See
# in_pre_hit_window()/would_hit() and game.py's calling code.
HAPTIC_LEAD_TIME_S = 0.5


def visual_lateral_distance(dx, dy):
    """Euclidean distance between two ball-space points, but scaled so
    it matches how far apart they'd actually LOOK on screen (see
    LATERAL_Y_SCREEN_SQUISH) rather than a plain world-space distance --
    use this (not raw sqrt(dx**2+dy**2)) for any "is this within an
    on-screen-circular radius" check, or a touch that's visually obvious
    in the y-direction can fail a check tuned by its x-direction
    appearance."""
    return (dx ** 2 + (dy * LATERAL_Y_SCREEN_SQUISH) ** 2) ** 0.5


class Ball:
    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        self.reset()

    def reset(self):
        """Spawns a new ball approaching from the far screen, with the
        target zone relocated to a new random spot for this shot."""
        self.x = self.rng.uniform(-0.2, 0.2)
        self.y = self.rng.uniform(-0.2, 0.2)
        self.z = 1.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = -APPROACH_SPEED
        self.state = "approaching"   # "approaching" | "returning" | "done"
        self.result = None           # None | "won" | "lost"

        # Decayed-peak-held swing vector (see track_swing()): magnitude
        # is the overall swing strength, x/y/z are the components
        # recorded at the moment that peak was set (not continuously
        # blended -- a swing's direction is whatever it was when it was
        # strongest, not some average of every direction touched along
        # the way).
        self._reset_swing_peak()
        self.new_target_zone()

    def _reset_swing_peak(self):
        self._swing_peak_magnitude = 0.0
        self._swing_peak_x = 0.0
        self._swing_peak_y = 0.0
        self._swing_peak_z = 0.0

    def new_target_zone(self):
        """Relocates the target zone to a new random spot."""
        center_x = self.rng.uniform(-TARGET_ZONE_CENTER_RANGE, TARGET_ZONE_CENTER_RANGE)
        center_y = self.rng.uniform(-TARGET_ZONE_CENTER_RANGE, TARGET_ZONE_CENTER_RANGE)
        self.target_zone = {
            "x_min": center_x - TARGET_ZONE_HALF_SIZE, "x_max": center_x + TARGET_ZONE_HALF_SIZE,
            "y_min": center_y - TARGET_ZONE_HALF_SIZE, "y_max": center_y + TARGET_ZONE_HALF_SIZE,
        }

    def bounce_off_wall(self):
        """Call once a returning shot has landed in the target zone
        (state "done" at z=1): instead of ending the rally, the ball
        reflects off the far wall back toward the player at
        WALL_BOUNCE_SPEED_FRACTION of its incoming speed -- depth
        velocity reversed, lateral velocity kept (a plain wall
        reflection), all scaled by the same fraction -- and becomes the
        next ball to hit. The target zone is NOT moved here; the caller
        decides when (see game.py), so the scored box stays on screen
        long enough to see that it scored."""
        speed = max(MIN_BOUNCE_APPROACH_SPEED, abs(self.vz) * WALL_BOUNCE_SPEED_FRACTION)
        self.vx *= WALL_BOUNCE_SPEED_FRACTION
        self.vy *= WALL_BOUNCE_SPEED_FRACTION
        self.vz = -speed
        self.z = 1.0
        self.state = "approaching"
        self.result = None
        self._reset_swing_peak()

    def update(self, dt):
        """Advances ball position. Returns True if still in play (caller
        should stop calling update() and read .result once False)."""
        if self.state == "returning" and self.z + self.vz * dt >= 1.0:
            # A fast return (high vz, from a hard swing) can jump past
            # z=1.0 in a single step -- landing wherever x/y end up AFTER
            # that full step, without interpolating, would place the
            # ball somewhere it never actually visually passed through.
            # Interpolate the exact point within this step where the
            # ball crosses z=1.0 instead, so the final resting position
            # (used for the landing marker/trail, and by game.py's own
            # pixel-space score check) is always where the ball's path
            # actually crossed the back wall, regardless of speed.
            frac = (1.0 - self.z) / (self.vz * dt) if self.vz * dt != 0 else 1.0
            frac = max(0.0, min(1.0, frac))
            self.x += self.vx * dt * frac
            self.y += self.vy * dt * frac
            self.z = 1.0

            # Whether this counts as landing in the target zone is
            # decided by game.py, in actual screen pixels, from this
            # landing position -- see game.py's lands_in_target_zone_px().
            # Default to "lost" here; game.py overrides to "won".
            self.state = "done"
            self.result = "lost"
            return False

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        if self.state == "approaching" and self.z <= 0:
            self.state = "done"
            self.result = "lost"

        return self.state != "done"

    def in_hit_window(self):
        """True if the ball is currently close enough in depth to attempt
        a hit. Deliberately does NOT also check self.state here: update()
        flips state to "done"/"lost" the instant z crosses to <= 0, and on
        that exact frame the caller's own "was it approaching before this
        update()" check is what should still gate the hit attempt -- if
        this also required state == "approaching", the single frame where
        z crosses from positive to <= 0 would silently skip the hit
        attempt entirely, letting the ball visually reach the racquet's
        circle without ever getting a chance to bounce off it."""
        return abs(self.z) <= HIT_ZONE_DEPTH

    def in_pre_hit_window(self):
        """True for a short window just before in_hit_window() opens --
        roughly HAPTIC_LEAD_TIME_S before the ball reaches the racquet.
        Only used to decide when it's time to check would_hit() and
        possibly fire the haptic wack early (to compensate for its own
        motor-pulse execution latency); it never decides whether a hit
        actually happens -- that's still try_hit() at the real contact
        moment."""
        # Uses the ball's actual approach speed, not APPROACH_SPEED: a
        # ball bounced back off the wall (bounce_off_wall()) can be
        # coming in much faster than a freshly served one.
        lead_depth = HIT_ZONE_DEPTH + abs(self.vz) * HAPTIC_LEAD_TIME_S
        return self.state == "approaching" and HIT_ZONE_DEPTH < self.z <= lead_depth

    def would_hit(self, racquet_x, racquet_y):
        """Non-mutating prediction: would a hit register right now,
        based purely on lateral position -- the same gate try_hit() uses,
        minus the depth/timing window (since this is meant to be checked
        slightly before the ball actually arrives). Does NOT check
        AprilTag detection status at all, by design: whatever
        racquet_x/racquet_y the caller currently has (frozen or
        dead-reckoned through a tag-loss gap) is honored exactly like
        try_hit() does -- a lost tag never blocks a hit on its own. Used
        only to decide whether to fire the haptic wack early; the real
        hit/bounce physics are still decided by try_hit() itself at the
        actual contact moment."""
        lateral_distance = visual_lateral_distance(self.x - racquet_x, self.y - racquet_y)
        return lateral_distance <= HIT_DISTANCE_MAX

    def track_swing(self, swing_x, swing_y, swing_z, dt):
        """Feeds in this frame's swing vector -- swing_x: left(-)/right(+),
        swing_y: down(-)/up(+), swing_z: away(-)/toward(+) the ball, all
        in the same units as the old scalar swing_speed (so a pure
        forward push alone produces the same magnitude as before) -- and
        keeps a decayed peak of it (see SWING_SPEED_HALF_LIFE). Call this
        every frame while the ball is approaching.

        Unlike a plain per-axis peak hold, the direction is only
        captured fresh AT THE MOMENT a new overall-magnitude peak is
        set; otherwise the whole held vector (direction included, not
        just its magnitude) decays together, shrinking toward zero
        without changing orientation -- so a swing that peaks a few
        frames before the exact contact frame still steers the ball by
        the direction it was actually swung in, not some blend of
        whatever the racquet did before and after, and its lateral kick
        fades along with its power instead of staying pinned at full
        strength forever."""
        magnitude = (swing_x ** 2 + swing_y ** 2 + swing_z ** 2) ** 0.5
        decay = 0.5 ** (dt / SWING_SPEED_HALF_LIFE)
        decayed_peak = self._swing_peak_magnitude * decay
        if magnitude >= decayed_peak:
            self._swing_peak_magnitude = magnitude
            self._swing_peak_x = swing_x
            self._swing_peak_y = swing_y
            self._swing_peak_z = swing_z
        else:
            self._swing_peak_magnitude = decayed_peak
            self._swing_peak_x *= decay
            self._swing_peak_y *= decay
            self._swing_peak_z *= decay

    def try_hit(self, racquet_x, racquet_y):
        """Attempts to register a hit. Returns True if it succeeded.

        Purely position-gated: requires only that the ball is in its
        hit window and the racquet is laterally close enough to the
        ball (a coarse check -- see would_hit()/HIT_DISTANCE_MAX). No
        minimum swing is required to register contact at all -- the
        racquet's circle is meant to behave like a solid boundary the
        ball ricochets off of the instant it touches, even a motionless
        racquet (the outgoing speed is floored at MIN_OUTGOING_SPEED so
        it still visibly bounces back rather than just stopping dead).

        On success, the tracked peak swing vector (see track_swing())
        decides everything about the outgoing shot: its magnitude sets
        the outgoing speed, and exactly ONE axis of it -- whichever of
        back-and-forth (power), left/right, or up/down changed the most
        -- steers the ball; the other two contribute nothing. So a
        mostly-straight power swing isn't curved by incidental small
        sideways noise, and a deliberate sideways/vertical swing isn't
        drowned out by forward power. This makes the position check
        above the only thing AprilTag contributes to a hit at all --
        the swing itself, from the IMU, decides the rest."""
        if not self.in_hit_window():
            return False
        if not self.would_hit(racquet_x, racquet_y):
            return False

        self.vz = max(MIN_OUTGOING_SPEED, self._swing_peak_magnitude * OUTGOING_SPEED_SCALE)

        forward_change = abs(self._swing_peak_z)
        right_change = abs(self._swing_peak_x)
        up_change = abs(self._swing_peak_y)

        # Steer as a fraction of the forward speed (see MAX_STEER_OFFSET):
        # the ball covers z 0 -> 1 at vz, so vx = offset * vz lands it
        # exactly `offset` units sideways, whatever the shot's speed.
        magnitude = self._swing_peak_magnitude
        if right_change > 0 and right_change >= forward_change and right_change >= up_change:
            self.vx, self.vy = self._swing_peak_x / magnitude * MAX_STEER_OFFSET * self.vz, 0.0
        elif up_change > 0 and up_change >= forward_change:
            self.vx, self.vy = 0.0, self._swing_peak_y / magnitude * MAX_STEER_OFFSET * self.vz
        else:
            self.vx, self.vy = 0.0, 0.0

        self.state = "returning"
        return True
