# Virtual Ping Pong

A ball approaches on screen; hit it with the racquet -- the circle drawn
around it is an exact, to-scale picture of the real hit radius, so if
the ball visually touches the circle it's a hit, period: no minimum
swing is required (even a motionless touch still bounces back, just
slowly). In real mode the racquet's position comes **only from hand
tracking** (MediaPipe on the webcam); your **swing** decides what
happens once you're there. Swing speed (how hard the ball comes back) is
a 50/50 blend of the hand's own speed and the AprilTag's speed; swing
direction comes from which way your hand moved. Back-and-forth motion
is power; left-right motion steers left/right; up-down motion steers
up/down -- exactly one of those three wins (whichever changed the
most), so a straight power swing isn't curved by incidental noise (see
`ball.py`'s `try_hit()`/`track_swing()`). A swing's peak is held for a
short window around contact, decaying gradually -- so swinging slightly
*before* the ball arrives still counts.

The ball then flies back toward a green target box on the far wall.
**Scoring is decided only by where the ball lands**, the same spot the
landing X is drawn: if the ball's drawn circle touches the box there,
it's a point. While the ball is in the air, a ring on the far wall
shows where it's going to land -- green if that's inside the box, red
if not. After each point, the box stays put for a moment (filled green
for a point, red for a miss) before the next ball and the next box
appear, so you can always see why a shot did or didn't count. A small
3D-ish paddle preview shows the paddle's orientation (cosmetic only).

## Two input modes

`game.py` supports two interchangeable racquet input sources behind the
same `RacquetInput` interface, selected with a flag -- `game.py` and
`ball.py` never need to change based on which one is active:

### Simulated (default): mouse/keyboard

```bash
python3 game.py
```

- **Mouse X position** -> paddle roll (cosmetic tilt only, no effect on physics)
- **Mouse X velocity** -> swing left/right (steers the ball)
- **Mouse Y position** -> depth: below the on-screen line = in range to hit
- **Mouse Y velocity**, downward only -> swing forward/power (a fast
  downward flick = a fast forward swing, mirroring how a real swing both
  closes the distance *and* imparts hit speed at once)
- **Q / E (hold)** -> swing up/down (steers the ball) and cosmetic paddle
  pitch tilt, both spring back to 0 when released
- **Esc** or closing the window -> quit

### Real: Lego Education Double Motor + webcam + AprilTag

```bash
python3 game.py --real
python3 game.py --real --camera 1   # pick a different camera index
```

- **Connection**: `RealRacquetInput` connects to whichever Double Motor
  hub advertises first/closest over Bluetooth -- no connection card or
  specific serial/color needed (calls `lelib.doubleMotor.connect()` with
  no filter arguments). It retries automatically until a hub is found.
- **Racquet position** -> **hand tracking only** (`arm_tracker.py`,
  MediaPipe's `HandLandmarker`, reusing the model/library from this
  course's `MobileRobots_HW1` project). The AprilTag is no longer used
  for position at all: it dropped out regularly (off-frame, motion blur
  mid-swing) and the racquet visibly teleported every time it came
  back. The wrist position is lightly smoothed
  (`POSITION_SMOOTHING_ALPHA`) so tracker jitter and re-acquiring the
  hand glide instead of jumping. While the hand is lost, the racquet
  holds its last position and its circle turns grey. The model file
  (`models/hand_landmarker.task`) auto-downloads on first run if
  missing.
- **Swing speed (power)** -> `POSE_SPEED_WEIGHT` x the hand's speed +
  `TAG_SPEED_WEIGHT` x the AprilTag's speed (both 0.5 by default). Each
  is measured frame-to-frame from the camera: image-plane movement plus
  closing-in speed (the hand's on-screen size growing; the tag's
  solvePnP distance shrinking). If the tag isn't visible, the hand's
  speed is used at full weight instead of being halved.
- **Swing direction (steering)** -> the hand's velocity direction, so
  the shot goes the way you visibly moved. The tag's direction only
  stands in on frames where the hand isn't trackable.
- **AprilTag**: print **`racquet_apriltag.png`** (tag ID 0) at **50mm**
  per black square edge and stick it flat on the racquet, facing the
  camera (correct `APRILTAG_SIZE_MM` if printed at a different size). It
  supplements swing speed and drives the paddle preview's tilt, nothing
  else.
- **Paddle angle** -> cosmetic only (the paddle preview): the tag's
  visual pose when visible, falling back to the racquet hub's IMU
  orientation when it isn't. The hub's IMU no longer affects position
  or hit physics.
- **Live camera panel**: the side dashboard shows the camera feed
  (mirrored, selfie-style) with the tracked wrist circled in yellow (or
  "NO HAND DETECTED") -- that wrist is exactly what positions the
  racquet. The tag isn't outlined, since it doesn't position anything.
- **Calibration**: `ARM_VELOCITY_TO_SWING_SCALE`,
  `ARM_GROWTH_TO_SWING_SCALE` and `CLOSING_SPEED_TO_SWING_SPEED_SCALE`
  at the top of `real_racquet_input.py` depend on your camera and
  playing distance. Run `python3 calibrate_real_input.py` (add
  `--camera 1` etc. to match) to print live readings and tune them.
  In real mode, the terminal also prints a `[hit]` line per hit (with
  the pose and tag speeds that were blended) and a `[miss]` line per
  missed ball.
- **Ground-truth found on this hardware**: the raw hub's
  `imu.pitch`/`imu.roll` fields are swapped relative to our own
  pitch/roll convention for this particular mounting
  (`SWAP_IMU_PITCH_ROLL`) -- this only affects the paddle preview's
  IMU tilt fallback (cosmetic only).

Swapping input sources only means implementing `racquet_input.RacquetInput`'s
interface (`connect`/`update`/`disconnect`, exposing pitch/roll/yaw/
swing_speed/swing_dir_x/swing_dir_y/swing_dir_z/x/y/in_range); this is
exactly how `RealRacquetInput` was added alongside `SimulatedRacquetInput`
without touching `game.py`'s or `ball.py`'s core logic.

## Haptic feedback

`game.py` calls `connect_haptics()`/`disconnect_haptics()` alongside the
racquet input's own connect/disconnect, and `trigger_haptic_hit()` when
a hit happens. `haptics.py` drives a Double Motor hub (separate hub
from the racquet's own, connects to whichever advertises first/closest
-- set `HAMMER_CARD_SERIAL` if that risks ambiguity with the racquet's
hub; retries up to `HAPTIC_CONNECT_ATTEMPTS` times, 1s apart, before
giving up).

Bench-tested with `wacker_pwm_test.py`: a single `motor_run_for_degrees`
pulse (`WACK_DEGREES = -45` at `WACK_SPEED_PCT = 50`) is the whole wack
motion the hammer needs -- the torsion spring's own tension pulls it
back on its own afterward, no separate "wind it up" step at all. The
pulse runs on a **background thread**, so `trigger_haptic_hit()` returns
immediately and the game never waits on the hammer. (It used to block
the game loop until the hub reported the move complete -- and
legoeducation has no timeout on that wait, so whenever the spring
stalled the motor short of `WACK_DEGREES`, the whole game froze on the
hit.) Inside the worker, every side except the last is still sent
non-blocking and the last one blocking -- see `haptics.py`'s docstring
for why firing every side fully non-blocking silently drops a motor.
While a pulse is still in flight, further triggers are skipped.
`HAPTIC_COOLDOWN_S` enforces a minimum gap between wacks regardless of
caller, so a sustained signal (e.g. from
`test_haptics_swing_trigger.py`'s continuous polling) can't fire the
hammer more than once for what's physically one motion.

Hit detection is purely position-gated (see "Tuning" below) -- whatever
racquet position `real_racquet_input.py` currently reports from hand
tracking is honored as-is. The wack itself fires up to `ball.HAPTIC_LEAD_TIME_S`
*before* the ball actually reaches the racquet -- compensating for the
motor pulse's own execution latency, so the physical smack lands close
to the moment of visual contact instead of noticeably after it -- but
only if a hit would already register at that earlier point
(`ball.would_hit()`); if the racquet isn't in position yet that early,
it falls back to firing at the real moment of contact instead, same as
before.

If `HAMMER_MOTOR_SIDE = le.MOTOR_BOTH` (the bench-tested default, both
motors firing together), the pulse's degrees are mirrored for the right
side (same reasoning as `wacker_pwm_test.py`'s `MIRROR_RIGHT_SIDE`) so
two oppositely-mounted motors spin the same real-world way; change
`HAMMER_MOTOR_SIDE` to `le.MOTOR_LEFT`/`le.MOTOR_RIGHT` if the hammer is
single-motor.

If no hammer hub is found after all retry attempts, `connect_haptics()`
prints a warning and the game falls back to the original print-only
stub behavior -- it's always safe to run without the hammer attached.

### Bench-testing the wacker hammer motor

`wacker_pwm_test.py` is a standalone tool (no pygame, no game loop) for
bench-testing the torsion-spring hammer mechanism on a Double Motor hub
(with the hammer plugged into one side, or both) before wiring it into
`haptics.py`: an interactive prompt to try different duty cycles (raw
PWM, -100 to 100) and watch live position/power/speed telemetry to find
what holds the spring wound vs. what lets it creep back, plus commands
to wind a repeatable amount and to test the built-in hold/brake/coast
release behaviors. Run it with:

```bash
python3 wacker_pwm_test.py               # hammer on the left side (default)
python3 wacker_pwm_test.py --motor right # hammer on the right side
python3 wacker_pwm_test.py --motor both  # hammer driven by both sides together
```

See its module docstring for the full command list.

## Sound

`sound.py` synthesizes three short percussive tones procedurally (hit,
score, miss) via pygame's audio mixer -- no asset files needed, same
tone-generation idea as `ME_193_HW3/songs.py`.

## Live score (MQTT)

`score_mqtt.py` publishes the current score/misses in real time to the
public `test.mosquitto.org` broker, under `ME193/Rogers/Shanmugam` --
same broker and `mqttlib.py` wrapper (reused from `ME193HW2`) this
course's other assignments already use, just with this project's own
name instead of a team placeholder (see
`ME193_TwoComputer/config.py`'s `TEAM_NAME` convention). `game.py` calls
`connect_score_mqtt()`/`disconnect_score_mqtt()` alongside its other
optional connections, and publishes every time a point is won or lost.
If the broker is unreachable, it prints a warning and the game keeps
playing with no live publishing -- same "optional, never blocks the
game" philosophy as `haptics.py`. Subscribe to the topic (e.g.
`mosquitto_sub -h test.mosquitto.org -t ME193/Rogers/Shanmugam`) to
watch it live.

## Files

- `ball.py` -- pure ball physics/scoring logic (approach, hit-window,
  reflection physics, target-zone check). No pygame, no hardware, fully
  unit-testable.
- `racquet_input.py` -- the `RacquetInput` interface, plus
  `SimulatedRacquetInput` (mouse/keyboard).
- `real_racquet_input.py` -- `RealRacquetInput`: Double Motor IMU +
  webcam/AprilTag tracking, implementing the same `RacquetInput`
  interface.
- `apriltag_detector.py` -- AprilTag detection (reused from `ME193HW2`)
  plus `estimate_pose()`, pure pose math (corners in, pitch/roll/yaw/
  distance out) that's independently unit-testable without a camera.
- `racquet_apriltag.png` -- printable AprilTag (ID 0) to stick on the
  racquet/car for distance + lateral tracking and paddle-angle pose.
- `arm_tracker.py` -- MediaPipe hand/wrist tracking, described above
  (reused from this course's `MobileRobots_HW1` project).
- `models/hand_landmarker.task` -- MediaPipe's hand model, auto-downloaded
  on first `--real` run if missing (gitignored, not committed).
- `calibrate_real_input.py` -- standalone tool to print live sensor
  readings for tuning `real_racquet_input.py`'s calibration constants.
- `paddle_view.py` -- pure math for the 3D-ish paddle preview (a
  rotated + projected circular blade and rectangular handle, modeled as
  a real ping-pong paddle silhouette).
- `sound.py` -- procedural hit/score/miss tones.
- `haptics.py` -- the real hammer motor control described above.
- `score_mqtt.py` -- live score publishing over MQTT, described above.
- `mqttlib.py` -- small paho-mqtt wrapper, reused from `ME193HW2`.
- `wacker_pwm_test.py` -- standalone bench-test tool for the hammer
  motor's PWM/duty-cycle levels, separate from the game.
- `game.py` -- the pygame main loop tying everything together.

## Setup

This folder lives under a path containing a `:` (`MATLAB:Code For Tufts`),
and Python's `venv` refuses to create an environment inside a path with a
colon in it (colon is the `PATH` separator). So the virtual environment
lives outside that path, in your home directory:

```bash
python3 -m venv ~/.venvs/me193midterm
source ~/.venvs/me193midterm/bin/activate
pip install -r requirements.txt
```

`pygame` needs SDL2 (and its image/mixer/ttf sub-libraries) installed
first if `pip install` fails to build it:
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

## Run

```bash
source ~/.venvs/me193midterm/bin/activate
python3 game.py                   # simulated: mouse/keyboard
python3 game.py --real            # real: Double Motor + webcam/AprilTag
python3 game.py --real --camera 1 # pick a different camera index
```

For `--real` mode, calibrate first (see "Calibration" above):
```bash
python3 calibrate_real_input.py
```

## Tuning

Constants at the top of `ball.py`:
- `APPROACH_SPEED` -- how fast the ball closes in.
- `HIT_ZONE_DEPTH` -- how precise the timing window is.
- `HIT_DISTANCE_MAX` -- how far off-target (laterally) you can still be
  and register a hit -- a coarse position gate; it no longer requires
  any minimum swing on top of it, the racquet circle is a solid
  boundary the ball always ricochets off of. The racquet circle drawn
  in `game.py` (`HIT_RADIUS_PX`) is derived directly from this
  constant, so the visible circle always matches the real hit radius
  exactly.
- `HAPTIC_LEAD_TIME_S` -- how far before actual contact the haptic wack
  fires (see "Haptic feedback" above), to compensate for the motor
  pulse's own execution latency; doesn't affect scoring/bounce physics
  at all, only when the wack *fires*.
- `TARGET_ZONE_HALF_SIZE` -- the scoring region's fixed size.
- `MAX_STEER_OFFSET` -- how far (world units) a purely sideways or
  vertical swing moves the ball's landing spot away from where it was
  hit. Sized relative to the target box so a deliberate steer moves the
  shot across the box rather than off the screen.
- `TARGET_ZONE_CENTER_RANGE` -- how far the target zone's center can
  randomize each shot (it relocates every point).
- `BALL_RADIUS` -- the ball's own radius (world-space units, at z=1)
  used for the target-zone score check -- grazing the zone within this
  distance counts as landing in it. `game.py`'s `BALL_MIN_RADIUS` (the
  far-ball's drawn pixel radius) is derived directly from this, so the
  drawn ball always matches the real scoring radius exactly.
- `LATERAL_Y_SCREEN_SQUISH` -- `game.py` renders y-axis movement more
  compressed on screen than x (purely to keep the play area's vertical
  extent on screen), so a given y-distance produces fewer on-screen
  pixels than the same x-distance would. Every "is this close enough to
  count as touching" check (the racquet's hit circle, the target zone)
  corrects for this via `visual_lateral_distance()` so a vertical graze
  that's visually obvious on screen can't fail a check that only looks
  right for horizontal grazes. `game.py`'s `to_screen()` imports this
  same constant instead of hardcoding it, so rendering and scoring can
  never drift out of sync with each other again.
- `OUTGOING_SPEED_SCALE` -- how the tracked peak swing vector's
  magnitude/components translate into the ball's outgoing speed and
  steering.
- `MIN_OUTGOING_SPEED` -- floor so even a near-motionless touch still
  visibly bounces the ball back.
- `SWING_SPEED_HALF_LIFE` -- how long a momentum spike's "memory" lasts
  while holding the peak swing vector (direction included) through
  contact.

Constants at the top of `real_racquet_input.py`:
- `APRILTAG_SIZE_MM` -- the printed tag's real-world size; must match
  how you actually printed `racquet_apriltag.png`.
- `POSITION_SMOOTHING_ALPHA` -- how strongly each new wrist reading
  pulls the on-screen racquet toward it (1 = raw, lower = smoother but
  laggier).
- `POSE_SPEED_WEIGHT` / `TAG_SPEED_WEIGHT` -- the swing-speed blend
  (must sum to 1.0).
- `ARM_VELOCITY_TO_SWING_SCALE` / `ARM_GROWTH_TO_SWING_SCALE` /
  `CLOSING_SPEED_TO_SWING_SPEED_SCALE` -- camera calibration constants,
  see "Calibration" above.
- `SWAP_IMU_PITCH_ROLL` -- swaps which raw hub field (`imu.pitch` vs.
  `imu.roll`) feeds the paddle preview's pitch vs. roll while the tag is
  lost. See "Ground-truth found" above.
- `ANGLE_SMOOTHING_ALPHA` -- how much to smooth the paddle preview's
  angle frame-to-frame; lower if it feels laggy, raise if it still looks
  jittery.

Constants at the top of `racquet_input.py` tune the simulated mouse/
keyboard feel (pitch rate, spring-back speed, max tilt angles, the in-range
boundary line).
