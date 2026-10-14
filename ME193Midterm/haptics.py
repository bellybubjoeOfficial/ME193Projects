"""
haptics.py

Real haptic hammer control. Ground-truth bench testing with
wacker_pwm_test.py found that a single `wind -5` at 50% speed (i.e.
motor_run_for_degrees(-5, speed=50)) is the whole wack motion the
hammer needs: the torsion spring's own tension pulls it back again on
its own afterward, so there's no separate "wind it up first" step at
all -- the entire haptic action is this one pulse, fired at the exact
moment a real hit registers (trigger_haptic_hit()).

game.py calls connect_haptics()/disconnect_haptics() alongside its
racquet input's own connect/disconnect, and trigger_haptic_hit() at the
moment of contact; nothing else needs to know whether real hardware is
actually attached -- if connect_haptics() can't find a hub, it prints a
warning and trigger_haptic_hit() silently does nothing instead, so the
game stays playable without the hammer plugged in.

If HAMMER_MOTOR_SIDE is MOTOR_BOTH (the bench-tested configuration,
both motors firing together), the pulse's degrees are mirrored for the
right side (negated), same reasoning as wacker_pwm_test.py's
MIRROR_RIGHT_SIDE: two motors mounted facing opposite ways need
opposite commands to spin the same real-world way.

Sending two BLE motor commands fully non-blocking, back-to-back, can
silently drop the second one (the exact bug wacker_pwm_test.py hit
earlier in this project -- the second motor just never spins, no
error raised). The fix proven there, and reused here exactly: fire
every side except the last with blocking=False, and let the last one
block. That blocking call waits for the hub to report the move
COMPLETE, and legoeducation has no timeout on that wait -- if the
hammer/spring stalls the motor short of WACK_DEGREES, it never
completes and the caller hangs indefinitely. That's what froze the game
on every hit, so the whole pulse now runs on a background worker
thread (_wack_worker()): trigger_haptic_hit() only hands it off and
returns immediately, and the game loop never waits on the hub. While a
pulse is still in flight, further triggers are skipped rather than
queued up behind a possibly-stuck one.
"""

import threading
import time

import legoeducation as le

from lelib import doubleMotor

# Which side(s) of the Double Motor hub the hammer is plugged into.
# Change to le.MOTOR_LEFT or le.MOTOR_RIGHT if it's wired to only one
# side.
HAMMER_MOTOR_SIDE = le.MOTOR_BOTH

# Connection card serial to pin the hammer's hub to a specific card,
# instead of connecting to whichever Double Motor advertises first.
# Only needed if "closest/first advertising" risks grabbing the
# racquet's own Double Motor hub instead of the hammer's.
HAMMER_CARD_SERIAL = None

MIRROR_RIGHT_SIDE = True  # see module docstring

# How many times to try connecting before giving up and falling back to
# stub mode. Bounded (unlike real_racquet_input.py's racquet connect(),
# which retries forever) because the hammer is optional -- the game
# must not hang at startup forever if it's genuinely not plugged in.
# But a single attempt (the previous behavior) gave up instantly on any
# hiccup -- including the hammer's hub simply still finishing its
# power-on/advertising sequence a moment after the game started -- with
# no second chance for the rest of that session. This gives it a real,
# but bounded, window to be found.
HAPTIC_CONNECT_ATTEMPTS = 5
HAPTIC_CONNECT_RETRY_DELAY_S = 1.0

# Bench-tested in wacker_pwm_test.py (its `wind -5 50` command).
WACK_DEGREES = -45
WACK_SPEED_PCT = 50

# Minimum time between wacks, enforced here (not by callers) so it
# applies no matter who calls trigger_haptic_hit() -- the real game,
# test_haptics_swing_trigger.py, or anything else. A single swing's
# acceleration can stay above a caller's own trigger threshold for
# several consecutive readings, which without this would fire the
# hammer more than once for what's physically one swing.
HAPTIC_COOLDOWN_S = 0.5

_dm = None
_connected = False
_last_wack_time = -HAPTIC_COOLDOWN_S
_wack_in_flight = threading.Event()


def _sides():
    if HAMMER_MOTOR_SIDE == le.MOTOR_BOTH:
        return [le.MOTOR_LEFT, le.MOTOR_RIGHT]
    return [HAMMER_MOTOR_SIDE]


def _mirrored_degrees(degrees, side):
    if HAMMER_MOTOR_SIDE == le.MOTOR_BOTH and MIRROR_RIGHT_SIDE and side == le.MOTOR_RIGHT:
        return -degrees
    return degrees


def connect_haptics():
    """Connects to the hammer's Double Motor hub, retrying up to
    HAPTIC_CONNECT_ATTEMPTS times. On final failure (no hub found,
    etc.) prints a warning and leaves haptics in no-op stub mode
    -- the game still runs fine without it."""
    global _dm, _connected
    print("[haptics] Scanning for the closest/first advertising Double Motor "
          "(hammer)...")
    for attempt in range(1, HAPTIC_CONNECT_ATTEMPTS + 1):
        try:
            _dm = doubleMotor()
            _dm.connect(card_serial=HAMMER_CARD_SERIAL)
            _connected = True
            print("[haptics] Hammer connected.")
            return
        except Exception as exc:
            more = attempt < HAPTIC_CONNECT_ATTEMPTS
            print(f"[haptics] Connect attempt {attempt}/{HAPTIC_CONNECT_ATTEMPTS} failed "
                  f"({exc}){' -- retrying...' if more else ''}")
            if more:
                time.sleep(HAPTIC_CONNECT_RETRY_DELAY_S)

    print(f"[haptics] Could not connect to the hammer's Double Motor after "
          f"{HAPTIC_CONNECT_ATTEMPTS} attempts -- continuing without real haptics.")
    _dm = None
    _connected = False


def disconnect_haptics():
    global _connected
    if _dm is not None:
        try:
            sides = _sides()
            for i, side in enumerate(sides):
                blocking = i == len(sides) - 1
                _dm.motor_stop(motor=side, blocking=blocking)
            _dm.disconnect()
        except Exception as exc:
            print(f"[haptics] Error while disconnecting: {exc}")
    _connected = False


def _wack_worker():
    """Runs the actual BLE motor commands, off the game loop's thread --
    see module docstring for why this can block indefinitely."""
    sides = _sides()
    try:
        for i, side in enumerate(sides):
            blocking = i == len(sides) - 1  # last side blocks -- see module docstring
            degrees = _mirrored_degrees(WACK_DEGREES, side)
            _dm.motor_run_for_degrees(degrees, speed=WACK_SPEED_PCT, motor=side, blocking=blocking)
    except Exception:
        # Keep playing silently; future hits will keep trying in case it
        # was transient (dropped connection, out of range, etc.).
        pass
    finally:
        _wack_in_flight.clear()


def trigger_haptic_hit():
    """Fires the wack pulse without blocking the caller. Returns True if
    it actually fired (handed to the worker thread, or a no-op stub without hardware),
    False if suppressed -- too soon after the last one
    (HAPTIC_COOLDOWN_S), or the previous pulse hasn't finished yet."""
    global _last_wack_time
    now = time.monotonic()
    if now - _last_wack_time < HAPTIC_COOLDOWN_S:
        return False

    if not _connected:
        _last_wack_time = now
        return True
    if _wack_in_flight.is_set():
        return False
    _last_wack_time = now
    _wack_in_flight.set()
    threading.Thread(target=_wack_worker, name="haptic-wack", daemon=True).start()
    return True
