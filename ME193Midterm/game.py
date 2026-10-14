"""
game.py

Virtual ping pong: the ball approaches, you must hit it with the racquet
(close enough in depth and roughly close enough laterally -- the
AprilTag's position is only a coarse gate) before it reaches you. A hit
sends it back toward a target zone on the far screen -- land inside it
to win the point, outside (or miss the ball entirely) and you lose it.
The swing itself (the IMU's own acceleration -- back-and-forth for
power, left/right or up/down for steering) decides everything about how
the ball comes back; see ball.py's module docstring.

By default this uses SimulatedRacquetInput (mouse/keyboard). Pass --real
to use the actual Double Motor + camera/AprilTag racquet instead -- see
real_racquet_input.py. Either way game.py and ball.py don't change; see
racquet_input.py's docstring for why.

The right-hand side panel shows a live camera feed with the tracked
hand circled (real mode only) and the 3D paddle preview. The paddle's
drawn tilt is cosmetic (vision-derived orientation, for your own
situational awareness).

Controls (see racquet_input.py), simulated mode only:
    Mouse X (position) -> paddle roll (cosmetic tilt only)
    Mouse X (velocity) -> swing_dir_x (steer left/right)
    Mouse Y (position) -> depth (below the line = in range to hit)
    Mouse Y (velocity), down only -> swing_dir_z (forward power -- move
                     down fast to swing)
    Q / E (hold)  -> swing_dir_y (steer down/up) + cosmetic pitch tilt
    Esc / close window -> quit

Usage:
    python game.py                        # simulated (mouse/keyboard)
    python game.py --real                 # real Double Motor + camera
    python game.py --real --camera 1      # pick a different camera index
"""

import argparse
import sys
from collections import deque

import pygame

import paddle_view
from ball import BALL_RADIUS, HIT_DISTANCE_MAX, LATERAL_Y_SCREEN_SQUISH, Ball
from haptics import connect_haptics, disconnect_haptics, trigger_haptic_hit
from score_mqtt import SCORE_TOPIC, connect_score_mqtt, disconnect_score_mqtt, publish_score
from racquet_input import SimulatedRacquetInput
from sound import SoundEffects


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual ping pong")
    parser.add_argument("--real", action="store_true",
                         help="Use the real Double Motor + camera/AprilTag racquet "
                              "instead of mouse/keyboard.")
    parser.add_argument("--camera", type=int, default=0,
                         help="Camera device index (only used with --real).")
    return parser.parse_args()


SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 700
FPS = 60

SIDE_PANEL_WIDTH = 260
PLAY_AREA_WIDTH = SCREEN_WIDTH - SIDE_PANEL_WIDTH

PLAY_AREA_CENTER = (PLAY_AREA_WIDTH // 2, int(SCREEN_HEIGHT * 0.42))
PLAY_AREA_SCALE = 260          # pixels per unit of ball x/y at the far screen

# The far-ball radius is derived directly from ball.BALL_RADIUS (at z=1,
# where depth_scale is 0.4 -- see to_screen()) so the drawn ball always
# matches the radius target-zone scoring actually uses -- if the ball
# visually touches the target box, it counts as landing in it, and if it
# doesn't, it doesn't. BALL_MAX_RADIUS (at z=0, close to the racquet) is
# a separate, purely cosmetic choice -- nothing is scored at that end.
BALL_MIN_RADIUS = int(BALL_RADIUS * PLAY_AREA_SCALE * 0.4)  # ball radius (px) at z=1 (far)
BALL_MAX_RADIUS = 28                                        # ball radius (px) at z=0 (at the racquet)

HIT_LINE_Y = int(SCREEN_HEIGHT * 0.6)   # the horizontal line = "in range" boundary

# The racquet circle's radius is derived directly from ball.HIT_DISTANCE_MAX
# (at z=0, where depth_scale is always 1.0) so the drawn circle always
# matches the real hit-detection radius exactly -- if the ball visually
# touches this circle, it's within range to hit, and if it doesn't, it isn't.
HIT_RADIUS_PX = int(HIT_DISTANCE_MAX * PLAY_AREA_SCALE)

TRAIL_LENGTH = 15

RESULT_PAUSE_S = 1.2   # how long a finished point's result stays on screen (and, after a miss, before the next serve)

CAMERA_PANEL_RECT = pygame.Rect(PLAY_AREA_WIDTH + 15, 20, SIDE_PANEL_WIDTH - 30, 230)
PADDLE_PANEL_RECT = pygame.Rect(PLAY_AREA_WIDTH + 15, 270, SIDE_PANEL_WIDTH - 30, 230)
PREVIEW_PADDLE_SCALE = 70

COLOR_BG = (10, 12, 20)
COLOR_TARGET = (34, 197, 94)
COLOR_TARGET_FILL_WON = (20, 83, 45)
COLOR_TARGET_FILL_LOST = (90, 25, 25)
COLOR_BALL = (250, 204, 21)
COLOR_RACQUET_IN_RANGE = (34, 197, 94)
COLOR_RACQUET_OUT_RANGE = (136, 136, 136)
COLOR_TEXT = (230, 230, 230)
COLOR_PANEL_BG = (24, 24, 32)
COLOR_PADDLE_BLADE = (210, 40, 40)    # classic ping-pong paddle rubber red
COLOR_PADDLE_HANDLE = (130, 85, 45)   # wood
COLOR_LOST_MARKER = (220, 60, 60)
COLOR_HAND = (250, 204, 21)


def to_screen(x, y, z):
    """Maps ball-space (x, y in [-1,1], z in [0,1]) to screen pixels,
    with a simple linear perspective: things closer (z small) are larger
    and the lateral spread is a bit wider. +y is "up" (matching
    racquet_input.py's "invert so up = +y" convention), so it's
    SUBTRACTED here since pygame's own y-axis increases downward. The
    y-axis is additionally compressed by ball.LATERAL_Y_SCREEN_SQUISH
    (purely to keep the play area's vertical extent on screen) --
    imported from ball.py rather than redefined here so every "does this
    touch" check there stays provably in sync with what's actually
    drawn; see that constant's docstring."""
    depth_scale = 0.4 + 0.6 * (1.0 - z)  # 0.4x far, 1.0x at the racquet
    px = PLAY_AREA_CENTER[0] + x * PLAY_AREA_SCALE * depth_scale
    py = (PLAY_AREA_CENTER[1] + (1.0 - z) * (SCREEN_HEIGHT * 0.5)
          - y * PLAY_AREA_SCALE * depth_scale * LATERAL_Y_SCREEN_SQUISH)
    radius = BALL_MIN_RADIUS + (BALL_MAX_RADIUS - BALL_MIN_RADIUS) * (1.0 - z)
    return int(px), int(py), int(radius)


def target_zone_rect_px(target_zone):
    """Returns (x0, y0, x1, y1) in screen pixels for the target zone's
    rectangle at z=1, sorted so x0<=x1 and y0<=y1. Shared by drawing and
    by lands_in_target_zone_px() so the drawn box and the pixel-space
    score check can never disagree about where it actually is."""
    z = 1.0
    x0, y0, _ = to_screen(target_zone["x_min"], target_zone["y_min"], z)
    x1, y1, _ = to_screen(target_zone["x_max"], target_zone["y_max"], z)
    # y0/y1 may come back in either order now that +y renders as "up" --
    # build the rect from sorted bounds rather than assuming x0,y0 is the
    # top-left corner.
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    return left, top, right, bottom


def _circle_overlaps_rect_px(cx, cy, radius, x0, y0, x1, y1):
    closest_x = max(x0, min(cx, x1))
    closest_y = max(y0, min(cy, y1))
    return (cx - closest_x) ** 2 + (cy - closest_y) ** 2 <= radius ** 2


def lands_in_target_zone_px(x, y, target_zone):
    """True if a ball resting at (x, y) on the far wall (z=1) would
    visibly touch the target zone's drawn box -- checked in actual
    screen pixels, using the exact same to_screen()/target_zone_rect_px()
    the drawing does, so "the ball touches the box" and "it scored"
    always agree.

    Scoring is decided ONLY by where the ball lands (the same spot the
    landing X is drawn), not by whether its on-screen path happened to
    cross the box mid-flight: the box is drawn at the far wall, but a
    ball still in flight is drawn lower down the screen (perspective),
    so a mid-flight overlap counted shots that then visibly landed
    outside the box -- a green X sitting outside the target."""
    x0, y0, x1, y1 = target_zone_rect_px(target_zone)
    bx, by, bradius = to_screen(x, y, 1.0)
    return _circle_overlaps_rect_px(bx, by, bradius, x0, y0, x1, y1)


def predicted_landing(ball):
    """Where a returning ball will hit the far wall (z=1), extrapolated
    from its current velocity -- the ball flies in a straight line, so
    this is exact. None if it isn't returning."""
    if ball.state != "returning" or ball.vz <= 0:
        return None
    t = (1.0 - ball.z) / ball.vz
    return ball.x + ball.vx * t, ball.y + ball.vy * t


def draw_target_zone(screen, font, target_zone, flash=None):
    """flash: None, "won" or "lost" -- fills the box green/red while a
    point's result is being shown, so it's obvious whether that shot
    counted."""
    x0, y0, x1, y1 = target_zone_rect_px(target_zone)
    rect = pygame.Rect(x0, y0, x1 - x0, y1 - y0)
    if flash is not None:
        fill = COLOR_TARGET_FILL_WON if flash == "won" else COLOR_TARGET_FILL_LOST
        pygame.draw.rect(screen, fill, rect)
    pygame.draw.rect(screen, COLOR_TARGET, rect, width=3)
    label = font.render("TARGET", True, COLOR_TARGET)
    screen.blit(label, (x0, y0 - 22))


def draw_ball(screen, ball):
    if ball.state == "done":
        return
    px, py, radius = to_screen(ball.x, ball.y, max(0.0, min(1.0, ball.z)))
    pygame.draw.circle(screen, COLOR_BALL, (px, py), radius)


def draw_ball_trail(screen, trail):
    """A short fading trail of dots along the ball's return path, so
    where it flew is visible even after it's landed."""
    n = len(trail)
    for i, (x, y, z) in enumerate(trail):
        px, py, _ = to_screen(x, y, max(0.0, min(1.0, z)))
        t = (i + 1) / n  # 0..1, newer points closer to 1
        radius = max(2, int(4 * t))
        color = tuple(int(c * (0.2 + 0.8 * t)) for c in COLOR_BALL)
        pygame.draw.circle(screen, color, (px, py), radius)


def draw_landing_marker(screen, marker):
    """An X at the ball's final resting spot, green if it landed in the
    target zone and red otherwise -- shown briefly after each point."""
    if marker is None:
        return
    x, y, result = marker
    px, py, _ = to_screen(x, y, 1.0)
    color = COLOR_TARGET if result == "won" else COLOR_LOST_MARKER
    size = 10
    pygame.draw.line(screen, color, (px - size, py - size), (px + size, py + size), 3)
    pygame.draw.line(screen, color, (px - size, py + size), (px + size, py - size), 3)


def draw_predicted_landing(screen, ball):
    """A ring on the far wall where the returning ball is going to land,
    green if that's inside the target box, red if not -- so you can see
    whether a shot will score while it's still in the air."""
    landing = predicted_landing(ball)
    if landing is None:
        return
    x, y = landing
    px, py, radius = to_screen(x, y, 1.0)
    color = COLOR_TARGET if lands_in_target_zone_px(x, y, ball.target_zone) else COLOR_LOST_MARKER
    pygame.draw.circle(screen, color, (px, py), radius + 4, width=2)


def draw_racquet(screen, racquet_input):
    px, py, _ = to_screen(racquet_input.x, racquet_input.y, 0.0)
    color = COLOR_RACQUET_IN_RANGE if racquet_input.in_range else COLOR_RACQUET_OUT_RANGE
    pygame.draw.circle(screen, color, (px, py), HIT_RADIUS_PX, width=4)
    pygame.draw.line(screen, (60, 60, 70), (0, HIT_LINE_Y), (PLAY_AREA_WIDTH, HIT_LINE_Y), 1)


def draw_panel_frame(screen, font, rect, label):
    pygame.draw.rect(screen, COLOR_PANEL_BG, rect)
    pygame.draw.rect(screen, (70, 70, 80), rect, width=1)
    text = font.render(label, True, COLOR_TEXT)
    screen.blit(text, (rect.x + 6, rect.y + 4))
    return pygame.Rect(rect.x + 6, rect.y + 26, rect.width - 12, rect.height - 32)


def draw_camera_panel(screen, font, rect, racquet_input):
    content = draw_panel_frame(screen, font, rect, "Camera")

    if not hasattr(racquet_input, "last_frame_bgr"):
        msg = font.render("Simulated input -- no camera", True, (150, 150, 160))
        screen.blit(msg, msg.get_rect(center=content.center))
        return

    frame_bgr = racquet_input.last_frame_bgr
    if frame_bgr is None:
        msg = font.render("Waiting for camera...", True, (150, 150, 160))
        screen.blit(msg, msg.get_rect(center=content.center))
        return

    frame_h, frame_w = frame_bgr.shape[:2]
    frame_rgb = frame_bgr[:, ::-1, ::-1]  # mirror (selfie view) + BGR -> RGB
    surf = pygame.image.frombuffer(frame_rgb.tobytes(), (frame_w, frame_h), "RGB")
    surf = pygame.transform.smoothscale(surf, (content.width, content.height))
    screen.blit(surf, content.topleft)

    # Hand/arm tracker overlay (see arm_tracker.py) -- the hand is the
    # ONLY thing that positions the racquet, so it's the only thing
    # marked here (the AprilTag just supplements swing speed and isn't
    # drawn). wrist_x/y are normalized [0,1] in the ORIGINAL
    # (non-mirrored) frame, so mirror x to match the selfie view.
    arm_detection = getattr(racquet_input, "last_arm_detection", None)
    if arm_detection is not None:
        wx = content.x + (1.0 - arm_detection.wrist_x) * content.width
        wy = content.y + arm_detection.wrist_y * content.height
        pygame.draw.circle(screen, COLOR_HAND, (int(wx), int(wy)), 10, width=3)
    elif hasattr(racquet_input, "hand_seen"):
        msg = font.render("NO HAND DETECTED", True, COLOR_LOST_MARKER)
        screen.blit(msg, msg.get_rect(center=(content.centerx, content.y + 14)))


def draw_paddle_preview(screen, font, rect, racquet_input):
    content = draw_panel_frame(screen, font, rect, "Paddle preview")

    cx = content.centerx
    cy = content.centery

    shape = paddle_view.paddle_shape_2d(racquet_input.pitch, racquet_input.roll, racquet_input.yaw)

    def to_points(local_2d):
        return [(cx + px * PREVIEW_PADDLE_SCALE, cy + py * PREVIEW_PADDLE_SCALE) for px, py in local_2d]

    handle_points = to_points(shape["handle"])
    pygame.draw.polygon(screen, COLOR_PADDLE_HANDLE, handle_points, width=0)
    pygame.draw.polygon(screen, (10, 10, 15), handle_points, width=2)

    blade_points = to_points(shape["blade"])
    pygame.draw.polygon(screen, COLOR_PADDLE_BLADE, blade_points, width=0)
    pygame.draw.polygon(screen, (10, 10, 15), blade_points, width=2)


def draw_hud(screen, font, score, misses, status_message, mqtt_message, help_text_str):
    hud = font.render(f"Score: {score}   Misses: {misses}", True, COLOR_TEXT)
    screen.blit(hud, (16, 16))
    if status_message:
        msg = font.render(status_message, True, COLOR_TEXT)
        screen.blit(msg, (16, 44))
    if mqtt_message:
        msg = font.render(mqtt_message, True, (150, 150, 160))
        screen.blit(msg, (16, 68))
    help_text = font.render(help_text_str, True, (150, 150, 160))
    screen.blit(help_text, (16, SCREEN_HEIGHT - 28))


SIM_HELP_TEXT = "Mouse: swing (velocity steers/moves down to hit)   Q/E: swing up/down   Esc: quit"
REAL_HELP_TEXT = "Hand tracking moves the racquet; hand + AprilTag speed sets power   Esc: quit"


def main():
    args = parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Virtual Ping Pong")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    if args.real:
        from real_racquet_input import RealRacquetInput
        racquet_input = RealRacquetInput(camera_index=args.camera)
        help_text_str = REAL_HELP_TEXT
    else:
        racquet_input = SimulatedRacquetInput(SCREEN_WIDTH, SCREEN_HEIGHT)
        help_text_str = SIM_HELP_TEXT
    racquet_input.connect()
    connect_haptics()
    connect_score_mqtt()

    sound = SoundEffects()
    ball = Ball()
    trail = deque(maxlen=TRAIL_LENGTH)
    landing_marker = None
    haptic_fired_this_ball = False
    hit_this_ball = False
    # After each point the target box is filled green/red (and the
    # landing X drawn) for RESULT_PAUSE_S, so it's clear why the shot did
    # or didn't count. On a point won, the ball bounces straight back
    # off the wall into the next rally shot (ball.bounce_off_wall()) and
    # the box only relocates once that display ends (or as soon as the
    # bounced ball is hit, whichever comes first). On a point lost, play
    # pauses for the same time before a fresh ball is served.
    box_flash = None
    box_flash_timer = 0.0
    relocate_target_pending = False
    serve_pause_timer = 0.0

    score = 0
    misses = 0
    status_message = ""
    mqtt_message = ""
    status_timer = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        racquet_input.update(dt)

        if box_flash_timer > 0:
            box_flash_timer -= dt
            if box_flash_timer <= 0:
                box_flash = None
                landing_marker = None
                if relocate_target_pending:
                    ball.new_target_zone()
                    relocate_target_pending = False

        if serve_pause_timer > 0:
            serve_pause_timer -= dt
            if serve_pause_timer <= 0:
                ball = Ball()
                trail.clear()
                haptic_fired_this_ball = False
                hit_this_ball = False
        else:
            if ball.state == "approaching":
                ball.track_swing(racquet_input.swing_dir_x, racquet_input.swing_dir_y,
                                  racquet_input.swing_dir_z, dt)

            was_approaching = ball.state == "approaching"
            ball.update(dt)

            # Fire the haptic wack slightly before actual contact (see
            # ball.HAPTIC_LEAD_TIME_S) to compensate for the motor pulse's
            # own execution latency -- but only if a hit would ALREADY
            # register based on the racquet's current position. If the
            # racquet isn't there yet at this early point, this simply
            # doesn't fire here, and the real contact-time check below
            # still fires it as a fallback. trigger_haptic_hit() never
            # blocks (see haptics.py), so neither path can stall a frame.
            if ball.state == "approaching" and not haptic_fired_this_ball and ball.in_pre_hit_window():
                if ball.would_hit(racquet_input.x, racquet_input.y):
                    trigger_haptic_hit()
                    haptic_fired_this_ball = True

            if was_approaching and ball.in_hit_window():
                if ball.try_hit(racquet_input.x, racquet_input.y):
                    hit_this_ball = True
                    sound.play_hit()
                    if not haptic_fired_this_ball:
                        trigger_haptic_hit()
                        haptic_fired_this_ball = True
                    if relocate_target_pending:
                        # The bounced ball was hit before the last
                        # point's result display ended -- the new shot
                        # needs its new target now.
                        ball.new_target_zone()
                        relocate_target_pending = False
                        box_flash = None
                        box_flash_timer = 0.0
                        landing_marker = None

            if ball.state == "returning":
                trail.append((ball.x, ball.y, ball.z))

            # Re-derive from the ball's CURRENT state, not update()'s own
            # verdict from earlier this frame: update() can flip state to
            # "done"/"lost" the instant z crosses to <= 0, but try_hit()
            # just above gets a chance to override that right back to
            # "returning" on that same frame (see ball.in_hit_window()'s
            # docstring).
            if ball.state == "done":
                # Decided purely by where the ball landed -- the same spot
                # the landing X is drawn (see lands_in_target_zone_px()).
                won = hit_this_ball and lands_in_target_zone_px(ball.x, ball.y, ball.target_zone)
                if hit_this_ball:
                    landing_marker = (ball.x, ball.y, "won" if won else "lost")
                    box_flash = "won" if won else "lost"
                    box_flash_timer = RESULT_PAUSE_S
                if won:
                    score += 1
                    sound.play_score()
                    status_message = "SCORE! In the box."
                    print(f"[score] In the box -- score {score}, misses {misses}")
                    ball.bounce_off_wall()
                    trail.clear()
                    relocate_target_pending = True
                    haptic_fired_this_ball = False
                    hit_this_ball = False
                else:
                    misses += 1
                    sound.play_miss()
                    status_message = "MISS -- outside the box." if hit_this_ball else "MISS -- didn't hit the ball."
                    print(f"[miss] {'Outside the box' if hit_this_ball else 'Ball not hit'} -- "
                          f"score {score}, misses {misses}")
                    serve_pause_timer = RESULT_PAUSE_S
                if publish_score(score, misses):
                    mqtt_message = f"Sent to MQTT ({SCORE_TOPIC})"
                    print(f"[mqtt] Sent score={score} misses={misses} to {SCORE_TOPIC}")
                else:
                    mqtt_message = "MQTT offline -- score not sent"
                status_timer = RESULT_PAUSE_S

        status_timer -= dt
        if status_timer <= 0:
            status_message = ""
            mqtt_message = ""

        screen.fill(COLOR_BG)
        draw_target_zone(screen, font, ball.target_zone, flash=box_flash)
        draw_ball_trail(screen, trail)
        draw_ball(screen, ball)
        draw_predicted_landing(screen, ball)
        draw_landing_marker(screen, landing_marker)
        draw_racquet(screen, racquet_input)
        draw_camera_panel(screen, font, CAMERA_PANEL_RECT, racquet_input)
        draw_paddle_preview(screen, font, PADDLE_PANEL_RECT, racquet_input)
        draw_hud(screen, font, score, misses, status_message, mqtt_message, help_text_str)
        pygame.display.flip()

    racquet_input.disconnect()
    disconnect_haptics()
    disconnect_score_mqtt()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
