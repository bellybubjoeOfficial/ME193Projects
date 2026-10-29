"""Stage 3 recording: CSV writer, file naming, session log, request/status files.

How a recording is started (the App runs in a container, so scripts on the Q
talk to it through small JSON files under data/control/):

    scripts/record.py  --writes-->  data/control/request.json
    python/main.py     --polls it, counts down, records data/raw/<name>.csv,
                         appends a row to data/sessions.csv, and keeps
                         data/control/status.json up to date
    scripts/record.py  --reads status.json to show progress

Everything here is plain Python (no Arduino imports) so it's tested in pytest.
"""

import csv
import json
import os
import re
import threading
import time
from pathlib import Path

from wearable import config, imu

REQUEST_NAME = "request.json"
STOP_NAME = "stop.json"
LIVE_NAME = "live.json"            # Stages 6-7: start live inference
STATUS_NAME = "status.json"
LAST_CHOICE_NAME = "collect.json"
# requested_s = what you asked for; seconds = what was actually recorded
# (shorter if you pressed ENTER to stop early: stopped_early = 1).
SESSION_COLUMNS = ["file", "activity", "session", "started_at", "requested_s", "seconds", "samples",
                   "dropped", "stopped_early", "note"]


class CsvRecorder:
    """Writes samples to a CSV until `seconds` of MCU time have passed, then stops.

    Writes to <name>.csv.partial and renames when complete, so a half-written
    recording (power bank unplugged mid-way) is never mistaken for a finished one.
    Thread-safe: add() is called from the Bridge thread.
    """

    def __init__(self, path, seconds):
        self.path = Path(path)
        self.partial = self.path.with_suffix(".csv.partial")
        self.seconds = seconds
        self.lock = threading.Lock()
        self.armed = False
        self.done = False
        self.file = None
        self.writer = None
        self.t0_us = None
        self.n = 0
        self.dropped = 0
        self.last_seq = None
        self.stopped_early = False

    def arm(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(self.partial, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(imu.CSV_COLUMNS)
            self.armed = True

    def add(self, row):
        """Add one sample row (imu.CSV_COLUMNS order). Returns True once finished."""
        with self.lock:
            if not self.armed or self.done:
                return self.done
            if self.t0_us is None:
                self.t0_us = row[1]
            # Stop BEFORE writing the sample that would exceed the duration, so
            # a 60 s recording holds exactly 6000 samples at 100 Hz. The cut is
            # half a period early so µs timestamp jitter can't let sample 6001 in.
            if row[1] - self.t0_us >= self.seconds * 1e6 - 0.5e6 / config.SAMPLE_RATE_HZ:
                self._finish()
                return True
            if self.last_seq is not None and row[0] > self.last_seq + 1:
                self.dropped += row[0] - self.last_seq - 1
            self.last_seq = row[0]
            self.writer.writerow(row)
            self.n += 1
            return False

    def elapsed_s(self):
        """Seconds recorded so far (from the sample count)."""
        return self.n / config.SAMPLE_RATE_HZ

    def stop(self):
        """Finish now (ENTER pressed in scripts/collect.py). Keeps what was recorded."""
        with self.lock:
            if self.armed and not self.done:
                self.stopped_early = True
                self._finish()

    def _finish(self):
        self.file.close()
        self.partial.replace(self.path)
        self.done = True


# ---- File naming + session numbers -------------------------------------------

def _name_re(activity):
    return re.compile(rf"^{config.FILE_PREFIX}_{re.escape(activity)}_s(\d+)_.*\.csv$")


def next_session_number(raw_dir, activity):
    """1 + the highest session number already used for this activity (1 if none).

    Counts .csv.partial files too, so an interrupted recording's number isn't reused.
    """
    pat = _name_re(activity)
    used = [int(m.group(1)) for p in Path(raw_dir).glob("*")
            if (m := pat.match(p.name.removesuffix(".partial")))]
    return max(used, default=0) + 1


def recording_filename(activity, session, when=None):
    """e.g. ankle_walk_s03_2026-10-28T14-05.csv (time = board clock, UTC on the Q)."""
    stamp = time.strftime("%Y-%m-%dT%H-%M", time.localtime(when))
    return f"{config.FILE_PREFIX}_{activity}_s{session:02d}_{stamp}.csv"


def parse_filename(name):
    """'ankle_walk_s03_….csv' -> ('walk', 3). Raises ValueError if it doesn't match."""
    m = re.match(rf"^{config.FILE_PREFIX}_([a-z]+)_s(\d+)_.*\.csv$", Path(name).name)
    if not m or m.group(1) not in config.RECORD_ACTIVITIES:
        raise ValueError(f"not a recording filename: {name}")
    return m.group(1), int(m.group(2))


# ---- sessions.csv ---------------------------------------------------------------

def read_sessions(path):
    """All rows of sessions.csv as dicts (strings), [] if there's no file yet."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_sessions(path, rows):
    """Rewrite the whole file (temp file + rename, so it's never half-written)."""
    path = Path(path)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SESSION_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SESSION_COLUMNS})
    os.replace(tmp, path)


def append_session(path, row):
    """Append one row (dict with SESSION_COLUMNS keys) to sessions.csv, writing the header if new.

    If the file has an older set of columns (e.g. from before stopped_early
    existed), it's rewritten with the current ones first; old rows get "" there.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != SESSION_COLUMNS:
            _write_sessions(path, read_sessions(path))
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SESSION_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SESSION_COLUMNS})


def update_session(path, file, **fields):
    """Change fields (e.g. note="strap slipped") of the row for `file`. Returns True if found."""
    rows = read_sessions(path)
    hit = False
    for row in rows:
        if row["file"] == file:
            row.update({k: str(v) for k, v in fields.items()})
            hit = True
    if hit:
        _write_sessions(path, rows)
    return hit


def expected_seconds(sessions_path, file):
    """Requested duration for the duration check; None if unknown or stopped early on purpose."""
    row = next((r for r in read_sessions(sessions_path) if r["file"] == file), None)
    if row is None or row.get("stopped_early") == "1" or not row.get("requested_s"):
        return None
    return float(row["requested_s"])


def discard_recording(raw_dir, sessions_path, file):
    """Delete a recording's CSV (and plot) and remove its sessions.csv row."""
    for p in (Path(raw_dir) / file, (Path(raw_dir) / file).with_suffix(".png")):
        p.unlink(missing_ok=True)
    rows = read_sessions(sessions_path)
    _write_sessions(sessions_path, [r for r in rows if r["file"] != file])


def usable_seconds(seconds, trim_s=config.TRIM_SECONDS):
    """Length left after trimming both ends."""
    return max(0.0, float(seconds) - 2 * trim_s)


def min_useful_seconds(trim_s=config.TRIM_SECONDS):
    """Shortest recording that still gives one feature window after trimming."""
    return 2 * trim_s + config.WINDOW_SAMPLES / config.SAMPLE_RATE_HZ


def summary(rows, activities=config.RECORD_ACTIVITIES, trim_s=config.TRIM_SECONDS):
    """Per activity: (activity, sessions, recorded minutes, usable minutes after trim)."""
    out = []
    for a in activities:
        secs = [float(r["seconds"] or 0) for r in rows if r["activity"] == a]
        out.append((a, len(secs), sum(secs) / 60, sum(usable_seconds(s, trim_s) for s in secs) / 60))
    return out


def format_summary(rows, **kw):
    lines = [f"  {'activity':<9} {'sessions':>8} {'minutes':>8} {'usable':>7}"]
    total = [0, 0.0, 0.0]
    for a, n, m, u in summary(rows, **kw):
        lines.append(f"  {a:<9} {n:>8} {m:>8.1f} {u:>7.1f}")
        total = [total[0] + n, total[1] + m, total[2] + u]
    lines.append(f"  {'total':<9} {total[0]:>8} {total[1]:>8.1f} {total[2]:>7.1f}")
    return "\n".join(lines)


# ---- Menu helpers (scripts/collect.py) --------------------------------------------

QUIT = "quit"


def parse_choice(text, default, activities=config.RECORD_ACTIVITIES):
    """Menu input -> activity name, QUIT, or None (not understood).

    Accepts a number (1-based), a name, or an unambiguous start of one
    ("ru" -> run; "s" is ambiguous: stand/sit), blank = default.
    """
    t = text.strip().lower()
    if t == "":
        return default
    if t in ("q", "quit", "exit"):
        return QUIT
    if t.isdigit():
        i = int(t) - 1
        return activities[i] if 0 <= i < len(activities) else None
    if t in activities:
        return t
    starts = [a for a in activities if a.startswith(t)]
    return starts[0] if len(starts) == 1 else None


def load_last_choice(control_dir, default=config.RECORD_ACTIVITIES[0]):
    try:
        a = json.loads((Path(control_dir) / LAST_CHOICE_NAME).read_text())["activity"]
        return a if a in config.RECORD_ACTIVITIES else default
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return default


def save_last_choice(control_dir, activity):
    _write_json_atomic(Path(control_dir) / LAST_CHOICE_NAME, {"activity": activity})


# ---- Request / status files ---------------------------------------------------

def _write_json_atomic(path, data):
    """Write to a temp file then rename, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def make_request(activity, seconds=config.DEFAULT_RECORD_SECONDS,
                 countdown=config.DEFAULT_COUNTDOWN_SECONDS, note=""):
    """Validated recording request (a dict). Raises ValueError on bad input."""
    if activity not in config.RECORD_ACTIVITIES:
        raise ValueError(f"unknown activity {activity!r}; choose from {config.RECORD_ACTIVITIES}")
    if not seconds > 0:
        raise ValueError("seconds must be > 0")
    if countdown < 0:
        raise ValueError("countdown must be >= 0")
    return {"activity": activity, "seconds": float(seconds), "countdown": float(countdown),
            "note": str(note), "requested_at": time.time()}


def write_request(control_dir, request):
    _write_json_atomic(Path(control_dir) / REQUEST_NAME, request)


def take_request(control_dir):
    """Read and remove a pending request. Returns the dict, or None if there isn't one.

    A malformed request is renamed to request.json.bad and raises ValueError.
    """
    path = Path(control_dir) / REQUEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        req = make_request(data["activity"], data.get("seconds", config.DEFAULT_RECORD_SECONDS),
                           data.get("countdown", config.DEFAULT_COUNTDOWN_SECONDS), data.get("note", ""))
    except (ValueError, KeyError, TypeError) as e:
        path.replace(path.with_suffix(".json.bad"))
        raise ValueError(f"bad request file: {e}") from e
    path.unlink()
    return req


def write_stop(control_dir):
    """Ask the App to stop the current recording (or cancel the countdown)."""
    _write_json_atomic(Path(control_dir) / STOP_NAME, {"requested_at": time.time()})


def take_stop(control_dir):
    """True (and removes the file) if a stop was requested."""
    path = Path(control_dir) / STOP_NAME
    if not path.exists():
        return False
    path.unlink(missing_ok=True)
    return True


def write_live_request(control_dir, model_name):
    """Ask the App to start live inference with "model1" or "model2" (Stages 6-7)."""
    if model_name not in config.LIVE_MODEL_FILES:
        raise ValueError(f"unknown model {model_name!r}; choose from {list(config.LIVE_MODEL_FILES)}")
    _write_json_atomic(Path(control_dir) / LIVE_NAME, {"model": model_name, "requested_at": time.time()})


def take_live_request(control_dir):
    """Read and remove a pending live request. Returns "model1"/"model2", or None.

    A malformed request is renamed to live.json.bad and raises ValueError.
    """
    path = Path(control_dir) / LIVE_NAME
    if not path.exists():
        return None
    try:
        name = json.loads(path.read_text())["model"]
        if name not in config.LIVE_MODEL_FILES:
            raise ValueError(f"unknown model {name!r}")
    except (ValueError, KeyError, TypeError) as e:
        path.replace(path.with_suffix(".json.bad"))
        raise ValueError(f"bad live request file: {e}") from e
    path.unlink()
    return name


def write_status(control_dir, state, **fields):
    """state: idle | countdown | recording | done | cancelled | error | live (+ any extra fields)."""
    _write_json_atomic(Path(control_dir) / STATUS_NAME,
                       {"state": state, "updated_at": time.time(), **fields})


def read_status(control_dir):
    path = Path(control_dir) / STATUS_NAME
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
