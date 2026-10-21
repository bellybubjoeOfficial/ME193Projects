"""
config.py

Every tunable number for the sumo wrestler in one place, so bench tests
only ever need to edit this file.
"""

# ── LEGO devices ────────────────────────────────────────────────────────────
# Connect only to the hubs that scanned OUR Connection Card (azure, 1096), not
# whichever hub advertises first -- other teams' hubs are nearby. One card
# pins all three devices (the library also filters by device type).
# Set a device's serial/color to None to fall back to "first one found".
LEGO_CARD_COLOR = "azure"   # card color by name: the light-blue card ("blue" is a different card)
LEGO_CARD_SERIAL = 1096

DRIVE_CARD_SERIAL = LEGO_CARD_SERIAL
DRIVE_CARD_COLOR = LEGO_CARD_COLOR

# Flip if hw_drive.py shows a side spinning the wrong way.
INVERT_LEFT = False
INVERT_RIGHT = False

# ── Xbox controller ─────────────────────────────────────────────────────────
STICK_DEADZONE = 0.12    # fraction of full stick travel ignored around center
TRIGGER_DEADZONE = 0.05

# ── Driving ─────────────────────────────────────────────────────────────────
MAX_SPEED = 100          # LEGO speed units (-100..100)
TURN_GAIN = 0.7          # <1 makes turning gentler than straight driving

# ── Downward color sensor (ring edge) ───────────────────────────────────────
COLOR_CARD_SERIAL = LEGO_CARD_SERIAL
COLOR_CARD_COLOR = LEGO_CARD_COLOR
# Named colors (lelib.colorSensor.detect_color) that count as ring tape.
# Run tests/hw_color.py over the tape and the floor to confirm these.
TAPE_COLOR_NAMES = ("Blue",)
EDGE_BLOCK_S = 1.5       # reverse stays blocked this long after the last tape reading
RUMBLE_STRENGTH = 1.0    # 0..1
RUMBLE_MS = 300

# ── Feet (crank legs on the Double Motor) ───────────────────────────────────
# Base-position arcs live in feet_calibration.json (tests/hw_feet_calibrate.py).
SETTLE_SPEED = 30        # speed used to roll each foot into its base arc
                         # (keep it slow enough that the foot can't skip
                         # over the arc between two 30 Hz ticks)
SETTLE_TARGET_FRACTION = 0.5   # settle into the center 50% of each base arc

# ── Whacker (Single Motor) ──────────────────────────────────────────────────
WHACKER_CARD_SERIAL = LEGO_CARD_SERIAL
WHACKER_CARD_COLOR = LEGO_CARD_COLOR
WHACK_SPEED = 100        # speed at full trigger (RT one way, LT the other)

# Auto-whack: arm rests at BASE; opponent seen by the UNO Q → one STRIKE
# sweep to the far end; then back to base.
#   strike end = center - WHACKER_STRIKE_DEG ... center ... base = center + WHACKER_BASE_DEG
# (center = where the calibration sweep started, saved in whacker_calibration.json)
WHACK_TRIGGER_MM = 120   # something within this distance starts one strike
WHACK_REARM_MM = 250     # must back off past this before another strike
WHACKER_BASE_SIDE = 1    # +1 / -1: flip if base ends up on the wrong (left) side
WHACKER_BASE_DEG = 90    # base = this far from center
WHACKER_STRIKE_DEG = 90  # strike sweeps to this far on the OTHER side of center
WHACKER_SPEED = 100      # 0..100 for strike + return (hub decelerates on its own)
WHACKER_MANUAL_HOLD_S = 2.0   # after RT/LT, wait this long before going back to base
WHACKER_MOVE_TIMEOUT_S = 1.0  # give up on a move after this long (arm blocked)
WHACKER_TOL_DEG = 5      # "arrived" / "at base" = within this many degrees
WHACKER_MANUAL_SPEED_STEP = 10  # RT/LT speed is rounded to this (fewer BLE commands)
# RT/LT can never move the arm past base or the strike end (hub-side limits).

# Blind zones: whacker angles where it sits in front of the distance sensor
# (tests/hw_whacker_calibrate.py → whacker_calibration.json).
WHACKER_BLIND_DETECT_MM = 300   # during the sweep, a reading this close = "sees the whacker"
WHACKER_BLIND_MARGIN_DEG = 8    # extra safety margin added to each side of a blind zone

# ── UNO Q sensors over MQTT (see MQTT_CONTRACT.md) ──────────────────────────
MQTT_BROKER = "10.247.137.61"   # Mosquitto runs on the UNO Q
MQTT_PORT = 1883
SENSOR_STALE_S = 0.3     # no message for this long → reading is unknown

# ── Main loop ───────────────────────────────────────────────────────────────
LOOP_HZ = 30
RESEND_S = 0.5           # resend an unchanged motor command this often (keep-alive)
