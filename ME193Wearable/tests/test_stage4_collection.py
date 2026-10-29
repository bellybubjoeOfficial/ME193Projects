"""Stage 4 tests: LED matrix frames, early stop, sessions.csv edits, summary,
menu parsing, trimming, and the App's recording state machine (python/main.py
run against a fake Bridge). No hardware needed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from conftest import make_still_capture
from wearable import config, dataset, display, features, imu, recording, synth

MAIN_PY = Path(__file__).resolve().parent.parent / "python" / "main.py"


# ---- LED matrix frames --------------------------------------------------------

@pytest.mark.parametrize("frame", [display.idle(), display.countdown(10), display.recording(120, 0.5),
                                   display.done(), display.error(), display.countdown(1000)])
def test_frames_are_104_values_0_to_7(frame):
    assert len(frame) == 104 and all(0 <= v <= 7 for v in frame)
    assert len(display.to_bytes(frame)) == 104


def test_countdown_digits_drawn_centred():
    assert display.as_text(display.countdown(8)).splitlines()[1:6] == [
        ".....###.....",
        ".....#.#.....",
        ".....###.....",
        ".....#.#.....",
        ".....###.....",
    ]


def test_three_digits_fit_and_big_numbers_clamp():
    rows = display.as_text(display.countdown(120)).splitlines()
    assert rows[1] == "..#..###.###."                         # tops of "1", "2", "0": 11 px wide
    assert rows[5] == ".###.###.###."
    assert display.countdown(1000) == display.countdown(999)


@pytest.mark.parametrize("fraction,lit", [(0, 0), (0.5, 6), (1, 13), (1.5, 13)])
def test_progress_bar(fraction, lit):
    f = display.recording(30, fraction)
    assert sum(1 for x in range(13) if f[6 * 13 + x]) == lit
    assert sum(1 for x in range(13) if f[7 * 13 + x]) == lit


def test_idle_is_one_dim_dot_and_to_bytes_validates():
    assert sum(1 for v in display.idle() if v) == 1
    with pytest.raises(ValueError):
        display.to_bytes([8] * 104)
    with pytest.raises(ValueError):
        display.to_bytes([0] * 103)


# ---- Early stop -------------------------------------------------------------------

def test_recorder_stop_keeps_what_was_recorded(tmp_path):
    df = make_still_capture(seconds=20)
    r = recording.CsvRecorder(tmp_path / "x.csv", 120)
    r.arm()
    for row in df.itertuples(index=False):
        r.add(list(row))
    r.stop()
    assert r.done and r.stopped_early and r.n == 2000
    assert len(imu.load_csv(tmp_path / "x.csv")) == 2000
    r.stop()                                                   # second stop: no-op
    assert recording.CsvRecorder(tmp_path / "y.csv", 1).stop() is None   # never armed: no-op


def test_stop_file_round_trip(tmp_path):
    assert recording.take_stop(tmp_path) is False
    recording.write_stop(tmp_path)
    assert recording.take_stop(tmp_path) is True
    assert recording.take_stop(tmp_path) is False


# ---- sessions.csv -----------------------------------------------------------------

OLD_HEADER = "file,activity,session,started_at,requested_s,seconds,samples,dropped,note\n"


def test_old_sessions_file_is_migrated(tmp_path):
    p = tmp_path / "sessions.csv"
    p.write_text(OLD_HEADER + "a.csv,sit,1,2026-10-27T13:38:52,60.0,60.0,6000,0,\"chair, feet flat\"\n")
    recording.append_session(p, {"file": "b.csv", "activity": "walk", "stopped_early": 1})
    rows = recording.read_sessions(p)
    assert p.read_text().splitlines()[0] == ",".join(recording.SESSION_COLUMNS)
    assert [r["file"] for r in rows] == ["a.csv", "b.csv"]
    assert rows[0]["note"] == "chair, feet flat" and rows[0]["stopped_early"] == ""
    assert rows[1]["stopped_early"] == "1"


def test_update_note_and_discard(tmp_path):
    p, raw = tmp_path / "sessions.csv", tmp_path / "raw"
    raw.mkdir()
    for f in ("a.csv", "b.csv"):
        (raw / f).write_text("x")
        recording.append_session(p, {"file": f, "activity": "walk"})
    (raw / "a.png").write_text("x")
    assert recording.update_session(p, "b.csv", note="strap slipped")
    assert not recording.update_session(p, "zzz.csv", note="nope")
    recording.discard_recording(raw, p, "a.csv")
    assert not (raw / "a.csv").exists() and not (raw / "a.png").exists() and (raw / "b.csv").exists()
    assert [(r["file"], r["note"]) for r in recording.read_sessions(p)] == [("b.csv", "strap slipped")]


def test_expected_seconds(tmp_path):
    p = tmp_path / "sessions.csv"
    recording.append_session(p, {"file": "a.csv", "requested_s": 120, "stopped_early": 0})
    recording.append_session(p, {"file": "b.csv", "requested_s": 120, "stopped_early": 1})
    assert recording.expected_seconds(p, "a.csv") == 120
    assert recording.expected_seconds(p, "b.csv") is None           # short on purpose
    assert recording.expected_seconds(p, "c.csv") is None
    assert recording.expected_seconds(tmp_path / "missing.csv", "a.csv") is None


def test_summary_counts_minutes_and_usable_time():
    rows = [{"activity": "walk", "seconds": "120"}, {"activity": "walk", "seconds": "60"},
            {"activity": "run", "seconds": "8"}]
    s = {a: (n, m, u) for a, n, m, u in recording.summary(rows)}
    assert s["walk"] == (2, pytest.approx(3.0), pytest.approx((110 + 50) / 60))
    assert s["run"] == (1, pytest.approx(8 / 60), 0.0)               # shorter than the trim
    assert s["airbike"] == (0, 0, 0)
    text = recording.format_summary(rows)
    assert "airbike" in text and text.splitlines()[-1].split()[:2] == ["total", "3"]


def test_min_useful_length_is_trim_plus_one_window():
    assert recording.min_useful_seconds() == pytest.approx(2 * config.TRIM_SECONDS + 2.56)


# ---- Menu ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("", "walk"), ("  ", "walk"), ("3", "run"), ("5", "sit"), ("0", None), ("6", None),
    ("run", "run"), ("RU", "run"), ("b", None), ("air", "airbike"), ("s", None),
    ("st", "stand"), ("swim", None), ("q", recording.QUIT), ("quit", recording.QUIT),
])
def test_parse_choice(text, expected):
    assert recording.parse_choice(text, "walk") == expected


def test_last_choice_remembered(tmp_path):
    assert recording.load_last_choice(tmp_path) == config.RECORD_ACTIVITIES[0]
    recording.save_last_choice(tmp_path, "airbike")
    assert recording.load_last_choice(tmp_path) == "airbike"
    (tmp_path / recording.LAST_CHOICE_NAME).write_text("{broken")
    assert recording.load_last_choice(tmp_path) == config.RECORD_ACTIVITIES[0]


# ---- Trimming at processing time ------------------------------------------------------

def test_trim_removes_5s_each_end_without_touching_input():
    df = synth.make_recording("walk", seconds=120)
    t = dataset.trim(df)
    assert len(t) == 11000 and len(df) == 12000
    assert t["seq"].iloc[0] == df["seq"].iloc[500] and t.index[0] == 0


def test_trim_too_short_gives_empty_table():
    t = dataset.trim(synth.make_recording("walk", seconds=9))
    assert len(t) == 0 and list(t.columns) == imu.CSV_COLUMNS


def test_load_recordings_trims_filters_and_skips_missing(tmp_path):
    raw, sessions = tmp_path / "raw", tmp_path / "sessions.csv"
    raw.mkdir()
    for activity, secs in [("walk", 30), ("run", 30), ("sit", 30)]:
        name = f"ankle_{activity}_s01_x.csv"
        synth.make_recording(activity, secs).to_csv(raw / name, index=False)
        recording.append_session(sessions, {"file": name, "activity": activity})
    recording.append_session(sessions, {"file": "ankle_walk_s02_gone.csv", "activity": "walk"})

    recs = dataset.load_recordings(raw, sessions, activities=("walk", "run"))
    assert [(a, s) for _, a, s in recs] == [("walk", "ankle_walk_s01_x"), ("run", "ankle_run_s01_x")]
    assert all(len(df) == 2000 for df, _, _ in recs)                  # 30 s - 2 x 5 s
    ds = features.build_dataset(recs)
    assert len(ds) == 2 * len(features.window_starts(2000))


# ---- The App's state machine (python/main.py with a fake Bridge) ------------------------

class FakeBridge:
    def __init__(self):
        self.provided, self.sent = {}, []

    def provide(self, name, fn):
        self.provided[name] = fn

    def notify(self, name, *args):
        self.sent.append((name, args))

    def call(self, name, *args, timeout=None):
        self.sent.append((name, args))


class FakeLogger:
    def __init__(self, name):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    warning = info


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Fresh import of python/main.py, data folders in tmp_path, fake Bridge."""
    for name, sub in [("DATA_DIR", ""), ("CHECK_DIR", "check"), ("RAW_DIR", "raw"),
                      ("CONTROL_DIR", "control")]:
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "SESSIONS_CSV", tmp_path / "sessions.csv")
    bridge = FakeBridge()
    fake = types.ModuleType("arduino.app_utils")
    fake.Bridge, fake.Logger = bridge, FakeLogger
    fake.App = types.SimpleNamespace(run=lambda **kw: None)
    monkeypatch.setitem(sys.modules, "arduino", types.ModuleType("arduino"))
    monkeypatch.setitem(sys.modules, "arduino.app_utils", fake)
    spec = importlib.util.spec_from_file_location("wearable_main_under_test", MAIN_PY)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    main.bridge = bridge
    main.feed_state = {"seq": 0}
    return main


def feed(main, seconds):
    """Push `seconds` of still samples through the Bridge handler (raw counts)."""
    for _ in range(int(seconds * config.SAMPLE_RATE_HZ)):
        s = main.feed_state["seq"]
        main.on_imu(s, s * 10_000, 30, 2048, -40, 5, -3, 2)
        main.feed_state["seq"] = s + 1


def drawn(main):
    return [bytes(a[0]) for n, a in main.bridge.sent if n == "draw"]


def leds(main):
    return [a[0] for n, a in main.bridge.sent if n == "set_status"]


def test_app_records_a_full_session(app):
    recording.write_request(config.CONTROL_DIR, recording.make_request("walk", seconds=3, countdown=0))
    app.step_recording()
    assert recording.read_status(config.CONTROL_DIR)["state"] == "recording"
    feed(app, 3.5)
    app.step_recording()
    st = recording.read_status(config.CONTROL_DIR)
    assert st["state"] == "done" and st["samples"] == 300 and not st["stopped_early"]
    row, = recording.read_sessions(config.SESSIONS_CSV)
    assert row["activity"] == "walk" and row["session"] == "1" and row["stopped_early"] == "0"
    assert len(imu.load_csv(config.RAW_DIR / st["file"])) == 300
    assert leds(app)[-2:] == [app.LED_RECORDING, app.LED_DONE]
    assert drawn(app)[-1] == bytes(display.done())


def test_app_session_numbers_continue(app):
    for _ in range(2):
        recording.write_request(config.CONTROL_DIR, recording.make_request("run", seconds=1, countdown=0))
        app.step_recording()
        feed(app, 1.5)
        app.step_recording()
    assert [r["session"] for r in recording.read_sessions(config.SESSIONS_CSV)] == ["1", "2"]


def test_app_countdown_then_cancel(app):
    recording.write_request(config.CONTROL_DIR, recording.make_request("airbike", seconds=60, countdown=5))
    app.step_recording()
    st = recording.read_status(config.CONTROL_DIR)
    assert st["state"] == "countdown" and st["remaining_s"] == pytest.approx(5, abs=0.5)
    assert leds(app)[-1] == app.LED_COUNTDOWN and drawn(app)[-1] == bytes(display.countdown(5))
    recording.write_stop(config.CONTROL_DIR)
    app.step_recording()
    assert recording.read_status(config.CONTROL_DIR)["state"] == "cancelled"
    assert app.job is None and not list((config.RAW_DIR).glob("*.csv"))
    assert recording.read_sessions(config.SESSIONS_CSV) == []


def test_app_stop_early_keeps_data(app):
    recording.write_request(config.CONTROL_DIR, recording.make_request("run", seconds=120, countdown=0))
    app.step_recording()
    feed(app, 20)
    app.step_recording()
    assert drawn(app)[-1] == bytes(display.recording(100, 20 / 120))      # 100 s left, bar 1/6
    recording.write_stop(config.CONTROL_DIR)
    app.step_recording()
    st = recording.read_status(config.CONTROL_DIR)
    assert st["state"] == "done" and st["stopped_early"] and st["samples"] == 2000
    assert recording.read_sessions(config.SESSIONS_CSV)[0]["stopped_early"] == "1"
    assert recording.expected_seconds(config.SESSIONS_CSV, st["file"]) is None


def test_app_ignores_stale_stop_and_reports_bad_request(app):
    recording.write_stop(config.CONTROL_DIR)
    app.step_recording()                                                  # nothing running
    assert recording.take_stop(config.CONTROL_DIR) is False               # consumed, harmless
    (config.CONTROL_DIR / recording.REQUEST_NAME).write_text('{"activity": "swim"}')
    app.step_recording()
    assert recording.read_status(config.CONTROL_DIR)["state"] == "error"
    assert leds(app)[-1] == app.LED_ERROR and drawn(app)[-1] == bytes(display.error())
