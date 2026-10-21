# ME193 Sumo — Unit Testing Plan

We build the wrestler one sensor at a time. Each stage adds one sensor and
one new branch of the decision tree (`policy.py`). A stage is "done" only
when **both** kinds of test pass:

- **Logic tests (`test_*.py`)**: pure Python, no hardware, run in < 1 s.
  They feed fake sensor values into the code and check the result.
  Run them all with:
  `python -m unittest discover -s tests -t . -v` (from `ME193Sumo/`)
- **Hardware checks (`tests/hw_*.py`)**: short bench scripts that talk to
  the real device. Each one has a written pass criterion below; tick it off
  by hand.

Never start the next stage while a test from the current one is failing.

---

## Stage 1 — LEGO drive + Xbox controller

Decision tree v1 (inputs: controller only)

```
controller connected? ── no ──► STOP
        │ yes
B held (brake)? ── yes ──► STOP
        │ no
DRIVE  (left stick: up/down = throttle, left/right = turn)
```

| ID | Type | What | Pass when |
|----|------|------|-----------|
| U1.1 | logic | `xbox.state_from_raw`: normalizes sticks to -1..1, triggers to 0..1, applies deadzone, flips Y so up is + | `tests/test_xbox.py` green |
| U1.2 | logic | `drive.arcade_mix`: forward, reverse, spin, clamp to ±max | `tests/test_drive.py` green |
| U1.3 | logic | `policy.decide` v1: the three branches above | `tests/test_policy.py` green |
| H1.1 | hw | `tests/hw_xbox.py`: live printout of controller state | Sticks centered read 0.00; every stick, trigger and button moves the right field; unplugging/turning off shows `connected=False` |
| H1.2 | hw | `tests/hw_drive.py`: scripted moves on the Double Motor, no Xbox | Forward goes forward, spin-left turns left, it stops at the end. If not, flip `INVERT_LEFT/RIGHT` in `config.py` |
| H1.3 | hw | `main.py`: drive with the Xbox | Stick maps to the right direction; letting go stops within ~0.2 s; B brakes; turning the controller off stops the robot |

## Stage 2 — Downward Color Sensor (ring edge)  ✅ logic tests written

When the sensor first lands on the blue tape:
1. a double beep plays (`sounds.py`)
2. the Xbox controller rumbles (`xbox.rumble`)

And whenever it sees blue, **reverse is blocked** for `EDGE_BLOCK_S` = 1.5 s
after the last blue reading (so it stays blocked while parked on the tape).
Forward, forward arcs and turning in place are always allowed.

Tree v2 (`EdgeGuard` in `ring.py` remembers the last blue time):

```
controller connected? ── no ──► STOP
B held (brake)? ── yes ──► STOP
stick pulls back while reverse is blocked? ── yes ──► throttle zeroed (turning still allowed)
DRIVE from left stick
(alert = sound + rumble on the tick the tape is first seen, whatever branch wins)
```

| ID | Type | What | Pass when |
|----|------|------|-----------|
| U2.1 | logic | `ring.py`: `is_tape`, `EdgeGuard` (fires once per tape landing; reverse blocked while on tape and 1.5 s after the last blue reading) | `tests/test_ring.py` green |
| U2.2 | logic | Tree v2 one tick at a time: alert flag, reverse blocked, forward / arcs / turning allowed | `tests/test_policy.py` `TestTreeV2` green |
| U2.3 | logic | Scenario: guard + tree over simulated 30 Hz time, wired exactly like `main.py` (back onto tape → one alert, reverse stops, forward works, reverse returns 1.5 s after the last blue; driving forward onto tape keeps going forward) | `tests/test_policy.py` `TestEdgeScenario` green |
| H2.1 | hw | `tests/hw_color.py`: slide over ring surface, tape, outside floor | `TAPE?` is YES only on the tape, no flicker. Otherwise edit `TAPE_COLOR_NAMES` |
| H2.2 | hw | `tests/hw_alert.py`: press X | Hear the double beep and feel the rumble |
| H2.3 | hw | `main.py`: back slowly onto the tape, then forward | Beep + rumble once, robot stops; holding reverse does nothing until ~1.5 s after leaving the tape; forward works right away |

## Stage 3 — Feet base-position settling + Whacker on the triggers  ✅ logic tests written

The robot walks on crank feet. Each foot has a "base" encoder arc (flat
feet), found by calibration. When the stick is released, each foot keeps
turning in the direction it last turned until it's inside the **center 50%**
of its arc (`SETTLE_TARGET_FRACTION`), then stops. B (brake) and a lost controller stop immediately without settling.
If reverse is blocked by the edge guard, feet settle forward instead.

The whacker (Single Motor) swings one way on RT and the other on LT,
speed ∝ trigger pull, stop + hold on release.

Tree v3:

```
controller connected? ── no ──► STOP everything
B held? ── yes ──► STOP feet (no settling)
stick pulls back while reverse blocked? ── yes ──► throttle zeroed
stick in deadzone? ── yes ──► feet in base? ── yes ──► idle
                                  └─ no ──► roll each foot on in its last direction
DRIVE from left stick
(whacker = RT − LT on every connected branch)
```

| ID | Type | What | Pass when |
|----|------|------|-----------|
| U3.1 | logic | `feet.py`: arc contains/width incl. wrap past 0° and negative angles; `middle()` center sub-arc; marks + turned direction → arc (fixes the left-foot inversion); settle speed; calibration file round-trip | `tests/test_feet.py` green |
| U3.2 | logic | `whacker.py`: RT/LT opposite, proportional, cancel | `tests/test_whacker.py` green |
| U3.3 | logic | Tree v3 feet branch: idle, per-foot settling in last direction, brake/lost don't settle, uncalibrated → plain stop, reverse-blocked settles forward | `TestTreeV3Feet` green |
| U3.4 | logic | Whacker rides alongside driving/braking/settling, zero when controller lost | `TestTreeV3Whacker` green |
| U3.5 | logic | Scenario: walk backwards, release → feet keep turning backwards (never flip) until in base, then stay idle | `TestSettleScenario` green |
| H3.1 | hw | `tests/hw_feet_calibrate.py`: mark one end, turn *through* the base range, mark the other end → `feet_calibration.json`. `--verify` re-checks a saved file | Live readout says IN BASE / IN TARGET exactly when each foot is flat (both feet, not inverted); each range ≥ ~15° wide |
| H3.2 | hw | `python main.py --no-color --no-whacker`: walk, let go | Feet roll on the same way and stop near the middle of flat every time; B stops dead |
| H3.3 | hw | `tests/hw_whacker.py`: RT / LT | Opposite swings, speed follows trigger, holds on release. Note the end positions of a safe swing (future soft limits) |

## Stage 3b — Ring-out sad song (moved from the old Stage 2)

- `OUT_OF_RING` (latched) → STOP everything + play sad song
- Whacker soft limits from the H3.3 positions

Open question: if the floor inside and outside the tape is the same color,
the sensor alone can't tell inside from outside. Then we detect the
*crossing* (tape → floor while moving away from the centre) or mount the
sensor so it only crosses the tape when the robot is fully out.

## Stage 4 — Arduino UNO Q sensors over MQTT

Message format, topics and broker: **`MQTT_CONTRACT.md`** (single source
of truth for both sides). Mosquitto runs on the Q (`10.247.137.61:1883`);
the UNO Q code is written by a separate Claude session on the board.

### 4a — Receive sensor data ✅ done (H4.1 ALL PASS on the real Q: 19.9 Hz, 0 missed, 11 ms ping)

| ID | Type | What | Pass when |
|----|------|------|-----------|
| U4.1 | logic | `unoq_link.py`: parse/validate distance + imu + status, malformed → dropped (never crash), 0.3 s staleness, `mm: null` = unknown, seq-gap counting, publisher restart, rate | `tests/test_unoq_link.py` green |
| H4.0 | hw | Laptop pipeline with no Q: `tests/fake_unoq.py --broker test.mosquitto.org` + `tests/hw_unoq.py --broker test.mosquitto.org` | All checks pass except ping (public broker is ~300 ms, expected) ✅ done 2026-10-19 |
| H4.1 | hw | Real Q: `tests/hw_unoq.py` (add `--live` to watch distance) | ALL PASS: status online, ≥ 15 Hz, 0 malformed, < 5% missed, median ping < 100 ms |
| H4.2 | hw | Unplug / stop the Q publisher while `hw_unoq.py --live` runs | Distance shows `---` within 0.3 s; status goes `offline` (Last Will) |

### 4b — Strike from base on distance (tree v5)  ✅ logic tests written

**Blind zones:** at some angles the whacker sits in front of the distance
sensor. A calibration sweep (±90° around center) finds those absolute-encoder
angles (`whacker_calibration.json`, measured blind 234°→319°, center 275°);
while the whacker is in one (or its angle is unknown), the distance counts as
unknown, so it neither triggers nor re-arms.

**Arm positions:** strike end = center − 90° … center … base = center + 90°
(`WHACKER_BASE_SIDE` picks which side is base). Center's place on the motor's
relative counter is found from the absolute angle at startup, so the arm can
start anywhere. Moves are hub-side position control.

- startup: arm moves to **base** and holds
- something within `WHACK_TRIGGER_MM` (200 mm) → **one strike**: sweep through
  center to the far end; re-arm only past `WHACK_REARM_MM` (250 mm)
- after a strike → straight back to base
- RT/LT always override and cancel any move; after the driver lets go, the
  arm stays put for `WHACKER_MANUAL_HOLD_S` (2 s), then returns to base
- RT drives toward the strike end, LT toward base, as hub-side position
  moves to those end points, so the Xbox can **never** push the arm past
  either limit (same limits as the strike)

```
Whacker sub-tree:
controller connected? ── no ──► 0, cancel move
RT/LT pulled? ── yes ──► manual speed, cancel move
strike in progress? ── yes ──► keep going to strike end
opponent just came within 200 mm (and arm not blind)? ── yes ──► start STRIKE
returning to base? ── yes ──► keep going to base
away from base and no RT/LT for 2 s? ── yes ──► start HOME
hold
```

| ID | Type | What | Pass when |
|----|------|------|-----------|
| U4.2 | logic | `ProximityTrigger` (fires once on entering 200 mm, re-arm past 250 mm, noise near 200 doesn't retrigger, None never fires/re-arms) + `ArmGeometry` (base / strike end either side) + `center_relative` (incl. across 0° and after a counter reset) + center from calibration (saved or derived from old file = 275°) + `ArmMotion` (strike passes center to far end, home returns to base, blocked move times out, cancel) | `tests/test_whacker.py` green |
| U4.3 | logic | Whacker sub-tree: strike from base, continue, no restart mid-strike, strike while returning, go home when idle ≥ 2 s (and at startup), hold when manual < 2 s ago, hold at base, unknown position, manual override/cancel, controller lost | `TestTreeV5Whacker` green |
| U4.4 | logic | Scenario (simulated arm + time): center → base at startup; opponent parks at 150 mm → exactly one strike to the far end then straight back to base; re-arm → second strike; driver swings away → held 2 s → home; driver grabs mid-strike → held, no new strike, home after 2 s; unknown distance → none | `TestWhackerScenario` green |
| H4.3 | hw | `tests/hw_autowhack.py` (add `--xbox` for override): Q + whacker only | Arm goes to base (right side) on start; hand within 200 mm → one strike to the far end and straight back; no repeat while close; BLIND never triggers; with `--xbox`: triggers take over, arm waits ~2 s after release then returns to base. Flip `WHACKER_BASE_SIDE` if base is on the left |
| U4.7 | logic | `manual_move`: RT → strike end, LT → base, follows `WHACKER_BASE_SIDE`, speed rounded (never 0), wobble → same command, holding a trigger stops exactly at the end; scenario: holding RT then LT for 3 s each never leaves base..strike end | `TestManualMove` + `test_xbox_can_never_pass_the_end_points` green |
| U4.5 | logic | Blind zones from a simulated 180° sweep (`tests/test_whacker.py` `TestBlindZones`): finds the zone, margin, base/edges stay clear, zone crossing 0°, encoder counting either way, zone at the sweep turnaround, two zones, far readings ignored, unknown angle = blind, file round-trip | green |
| U4.6 | logic | Scenario: arm passing in front of the sensor never triggers; real opponent triggers when the arm is clear; a blind reading never re-arms | `TestBlindZoneScenario` green |
| H4.5 | hw | `tests/hw_whacker_calibrate.py`: whacker at base, nothing within 50 cm, sweeps base −90°→+90° and back, saves `whacker_calibration.json`; `--verify` shows BLIND live | Blind zone(s) match where the arm crosses the sensor; `--verify` says BLIND exactly there |
| H4.4 | hw | `main.py` everything on | Drive + auto-whack together; RT/LT still take over mid-whack |

## Stage 5 — Sound buttons  ✅ logic tests written

Y → sad tune ("wah wah wah wahhh", descending trombone with a wobbly last
note), A → victory fanfare (rising C–E–G–C). Fires once per press, not while
held; a new tune cuts off the one playing. Doesn't affect driving or the
whacker. The sad tune will be reused for the ring-out song.

| ID | Type | What | Pass when |
|----|------|------|-----------|
| U5.1 | logic | `ButtonEdges` (once per press, independent buttons, disconnect releases), tree sound branch (Y sad, A victory, held = silent, other buttons silent, doesn't change driving/whacker, works while braking), melody synthesis (length, no clipping, sad descends / victory rises, no clicks at ends) | `tests/test_sounds.py` green |
| H5.1 | hw | `tests/hw_alert.py`: press Y, then A | Hear the sad tune, then the fanfare; holding a button doesn't repeat it |
| H5.2 | hw | `main.py`: Y / A while driving | Tunes play; driving and the whacker are unaffected |

## Final checks

- Weight ≤ 1 kg including the UNO Q battery
- Fits in a 30 cm cube with the whacker at full extension
- 3 practice bouts with no crashes
