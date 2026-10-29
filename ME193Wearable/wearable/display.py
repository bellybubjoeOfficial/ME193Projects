"""Stage 4: pictures for the UNO Q's 8 x 13 LED matrix, built in Python.

A frame is a list of 104 brightness values 0..7, row by row (index = y*13 + x,
y = 0 is the top row). main.py sends it with Bridge.call("draw", bytes(frame))
and the sketch just copies it to the matrix. Building the pictures here keeps
the sketch tiny and lets pytest check them.

  idle        one dim dot in the bottom-right corner (App running, waiting)
  countdown   big seconds-left number
  recording   seconds left (dim) on top, progress bar filling the bottom 2 rows
  done        check mark
  error       X
  activity    Stage 6/7 live prediction: 3-letter name (SIT/STD/WLK/RUN/AIR) on top,
              confidence bar (vote fraction) on the bottom row, model number
              as 1 or 2 dots in the top-right corner
"""

ROWS, COLS = 8, 13
MAX = 7                     # brightest (3-bit grayscale)

# 3 x 5 digit font: each digit is 5 rows of 3 bits ("1" = LED on).
DIGITS = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
}


# 3 x 5 letters for the live activity labels.
LETTERS = {
    "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "D": ["110", "101", "101", "101", "110"],
    "I": ["111", "010", "010", "010", "111"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "N": ["101", "111", "111", "111", "101"],
    "R": ["110", "101", "110", "101", "101"],
    "S": ["011", "100", "010", "001", "110"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "W": ["101", "101", "101", "111", "101"],
    "?": ["111", "001", "010", "000", "010"],
}
ACTIVITY_TEXT = {"sit": "SIT", "stand": "STD", "walk": "WLK", "run": "RUN", "airbike": "AIR"}


def blank():
    return [0] * (ROWS * COLS)


def _set(frame, x, y, level=MAX):
    if 0 <= x < COLS and 0 <= y < ROWS:
        frame[y * COLS + x] = level


def _number(frame, n, top, level):
    """Draw integer n (0..999) centred horizontally, digit tops at row `top`."""
    text = str(max(0, min(999, int(n))))
    width = 4 * len(text) - 1                  # 3 px per digit + 1 px gaps
    left = (COLS - width) // 2
    for i, ch in enumerate(text):
        for dy, bits in enumerate(DIGITS[ch]):
            for dx, b in enumerate(bits):
                if b == "1":
                    _set(frame, left + 4 * i + dx, top + dy, level)


def _text(frame, text, top, level):
    """Up to 3 characters (digits or LETTERS), left-aligned at x = 0."""
    font = {**DIGITS, **LETTERS}
    for i, ch in enumerate(text[:3]):
        for dy, bits in enumerate(font.get(ch, LETTERS["?"])):
            for dx, b in enumerate(bits):
                if b == "1":
                    _set(frame, 4 * i + dx, top + dy, level)


def activity(label, confidence, model_number=2):
    """Live prediction: name on rows 1-5, confidence bar on row 7, model dots top-right."""
    f = blank()
    _text(f, ACTIVITY_TEXT.get(label, "?"), top=1, level=MAX)
    for x in range(round(max(0.0, min(1.0, confidence)) * COLS)):
        _set(f, x, 7, 3)
    for i in range(model_number):
        _set(f, COLS - 1, i * 2, 4)
    return f


def idle():
    f = blank()
    _set(f, COLS - 1, ROWS - 1, 2)
    return f


def countdown(seconds_left):
    f = blank()
    _number(f, seconds_left, top=1, level=MAX)
    return f


def recording(seconds_left, fraction_done):
    f = blank()
    _number(f, seconds_left, top=0, level=3)
    filled = round(max(0.0, min(1.0, fraction_done)) * COLS)
    for x in range(filled):
        _set(f, x, 6)
        _set(f, x, 7)
    return f


def done():
    f = blank()
    for x, y in [(2, 3), (3, 4), (4, 5), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2), (10, 1)]:
        _set(f, x, y)
    return f


def error():
    f = blank()
    for i in range(7):
        _set(f, 3 + i, 1 + i)
        _set(f, 9 - i, 1 + i)
    return f


def to_bytes(frame):
    if len(frame) != ROWS * COLS or not all(0 <= v <= MAX for v in frame):
        raise ValueError("frame must be 104 values in 0..7")
    return bytes(frame)


def as_text(frame):
    """ASCII picture for tests/debugging: '#' bright, '+' dim, '.' off."""
    chars = {0: "."}
    return "\n".join("".join(chars.get(v, "#" if v >= 5 else "+") for v in frame[y * COLS:(y + 1) * COLS])
                     for y in range(ROWS))
