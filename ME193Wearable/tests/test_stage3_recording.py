"""Stage 3 unit tests: recorder, file naming, session log, request files, validation.

No hardware needed: recordings are synthetic (conftest.make_still_capture).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_still_capture
from wearable import checks, config, imu, recording, validate
from wearable.checks import FAIL, PASS, WARN

SITTING_GRAVITY = (0.05, 0.95, 0.30)        # shin roughly vertical, tilted a bit
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def sit_df():
    g = np.array(SITTING_GRAVITY) / np.linalg.norm(SITTING_GRAVITY)
    return make_still_capture(seconds=60, gravity=tuple(g))


def rows(df):
    """Rows as lists with ints kept as ints (like imu.sample_row gives main.py)."""
    return [list(r) for r in df[imu.CSV_COLUMNS].itertuples(index=False)]


def record(df, path, seconds):
    """Feed df through a CsvRecorder like main.py does; returns the recorder."""
    r = recording.CsvRecorder(path, seconds)
    r.arm()
    for row in rows(df):
        if r.add(row):
            break
    return r


# ---- Recorder + file format ---------------------------------------------------

def test_recorder_writes_exact_columns_and_duration(tmp_path, sit_df):
    longer = make_still_capture(seconds=65)              # stream keeps going past the end
    r = record(longer, tmp_path / "x.csv", 60)
    assert r.done and (tmp_path / "x.csv").exists()
    assert not (tmp_path / "x.csv.partial").exists()
    df = imu.load_csv(tmp_path / "x.csv")
    assert list(df.columns) == imu.CSV_COLUMNS
    assert len(df) == 6000                                # 60 s x 100 Hz, not one more
    assert r.n == 6000 and r.elapsed_s() == pytest.approx(60)


def test_recorder_keeps_partial_name_until_finished(tmp_path, sit_df):
    r = recording.CsvRecorder(tmp_path / "x.csv", 60)
    r.arm()
    for row in rows(sit_df)[:100]:
        r.add(row)
    assert not r.done
    assert (tmp_path / "x.csv.partial").exists() and not (tmp_path / "x.csv").exists()


def test_recorder_ignores_samples_before_arm(tmp_path, sit_df):
    r = recording.CsvRecorder(tmp_path / "x.csv", 1)
    assert r.add(rows(sit_df)[0]) is False and r.n == 0


def test_recorder_counts_dropped_samples(tmp_path, sit_df):
    gappy = sit_df.drop(index=[100, 101, 102, 500]).reset_index(drop=True)
    r = record(gappy, tmp_path / "x.csv", 30)
    assert r.dropped == 4


def test_load_csv_rejects_wrong_columns(tmp_path, sit_df):
    p = tmp_path / "bad.csv"
    sit_df.rename(columns={"ax_g": "ax"}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        imu.load_csv(p)


# ---- File names + session numbers ---------------------------------------------

def test_filename_format():
    when = 1793196300          # 2026-10-28 14:05 UTC
    name = recording.recording_filename("walk", 3, when)
    assert name.startswith("ankle_walk_s03_2026-10-2") and name.endswith(".csv")
    assert recording.parse_filename(name) == ("walk", 3)


def test_session_numbers_increment_per_activity(tmp_path):
    assert recording.next_session_number(tmp_path, "sit") == 1
    for name in ["ankle_sit_s01_a.csv", "ankle_sit_s02_b.csv.partial", "ankle_walk_s07_c.csv", "notes.txt"]:
        (tmp_path / name).touch()
    assert recording.next_session_number(tmp_path, "sit") == 3          # partial s02 counts
    assert recording.next_session_number(tmp_path, "walk") == 8
    assert recording.next_session_number(tmp_path, "run") == 1


@pytest.mark.parametrize("name", ["ankle_swim_s01_x.csv", "walk_s01.csv", "ankle_walk_x.csv"])
def test_parse_filename_rejects_bad_names(name):
    with pytest.raises(ValueError):
        recording.parse_filename(name)


def test_sessions_csv_header_written_once(tmp_path):
    p = tmp_path / "sessions.csv"
    recording.append_session(p, {"file": "a.csv", "activity": "sit", "session": 1, "requested_s": 60})
    recording.append_session(p, {"file": "b.csv", "activity": "sit", "session": 2, "note": "strap, slipped"})
    s = pd.read_csv(p)
    assert list(s.columns) == recording.SESSION_COLUMNS
    assert list(s["file"]) == ["a.csv", "b.csv"]
    assert s["note"].iloc[1] == "strap, slipped"                         # comma survives quoting


# ---- Request / status files ---------------------------------------------------

def test_request_round_trip_removes_file(tmp_path):
    recording.write_request(tmp_path, recording.make_request("sit", 60, 10, "test"))
    req = recording.take_request(tmp_path)
    assert (req["activity"], req["seconds"], req["countdown"], req["note"]) == ("sit", 60, 10, "test")
    assert recording.take_request(tmp_path) is None                      # consumed


@pytest.mark.parametrize("kwargs", [dict(activity="swim"), dict(activity="sit", seconds=0),
                                    dict(activity="sit", countdown=-1)])
def test_make_request_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        recording.make_request(**kwargs)


def test_bad_request_file_is_set_aside(tmp_path):
    (tmp_path / recording.REQUEST_NAME).write_text(json.dumps({"activity": "swim"}))
    with pytest.raises(ValueError):
        recording.take_request(tmp_path)
    assert (tmp_path / "request.json.bad").exists()
    assert recording.take_request(tmp_path) is None                      # doesn't fail forever


def test_status_round_trip(tmp_path):
    assert recording.read_status(tmp_path) is None
    recording.write_status(tmp_path, "recording", elapsed_s=12.0)
    st = recording.read_status(tmp_path)
    assert st["state"] == "recording" and st["elapsed_s"] == 12.0


# ---- Validation checks ----------------------------------------------------------

def test_good_sitting_recording_passes_everything(sit_df):
    results = validate.run_all(sit_df, "sit", 60)
    assert [r.status for r in results] == [PASS] * len(results), [r for r in results if r.status != PASS]


def test_monotonic_catches_each_column(sit_df):
    assert validate.check_monotonic(sit_df).status == PASS
    for col, idx in [("seq", 50), ("t_mcu_us", 50), ("t_host_s", 50)]:
        bad = sit_df.copy()
        bad.loc[idx, col] = bad.loc[idx - 2, col]                         # step back in time
        r = validate.check_monotonic(bad)
        assert r.status == FAIL and col in r.detail


def test_host_time_may_repeat(sit_df):
    # Two samples arriving in the same Bridge packet can share a Linux timestamp.
    df = sit_df.copy()
    df.loc[10, "t_host_s"] = df.loc[9, "t_host_s"]
    assert validate.check_monotonic(df).status == PASS


def test_sample_rate_check_catches_wrong_rate_and_gaps(sit_df):
    slow = make_still_capture(seconds=60, fs=95)
    assert checks.check_sample_rate(slow).status == FAIL
    gappy = sit_df.drop(index=range(1000, 1050)).reset_index(drop=True)
    r = checks.check_sample_rate(gappy)
    assert r.status == FAIL and "dropped" in r.detail


def test_clipping_levels(sit_df):
    assert validate.check_clipping(sit_df).status == PASS
    few = sit_df.copy()
    few.loc[[100, 200], "ay_g"] = 15.9995                                 # int16 max at ±16 g
    assert validate.check_clipping(few).status == WARN
    many = sit_df.copy()
    many.loc[:600, "gz_dps"] = -2000.0                                    # 10 % pinned
    r = validate.check_clipping(many)
    assert r.status == FAIL and "gz_dps" in r.detail


def test_duration_check(sit_df):
    assert validate.check_duration(sit_df, 60).status == PASS
    assert validate.check_duration(sit_df.iloc[:5000], 60).status == FAIL


def test_sensible_values(sit_df):
    assert validate.check_sensible_values(sit_df, "sit").status == PASS
    counts = sit_df.copy()
    counts[imu.ACCEL_COLS] *= 2048                                        # forgot to convert to g
    assert validate.check_sensible_values(counts, "sit").status == FAIL
    fidget = sit_df.copy()
    fidget["gx_dps"] += 40 * np.sin(np.arange(len(fidget)) / 20)
    assert validate.check_sensible_values(fidget, "sit").status == WARN
    assert validate.check_sensible_values(fidget, "walk").status == PASS   # motion is fine there


# ---- The validate script end to end ---------------------------------------------

def test_validate_script_on_a_recorded_file(tmp_path, sit_df):
    raw = tmp_path / "raw"
    name = recording.recording_filename("sit", 1)
    r = record(make_still_capture(seconds=61, gravity=SITTING_GRAVITY), raw / name, 60)
    recording.append_session(tmp_path / "sessions.csv",
                             {"file": name, "activity": "sit", "session": 1, "requested_s": 60,
                              "seconds": r.elapsed_s(), "samples": r.n})
    out = subprocess.run([sys.executable, str(SCRIPTS / "validate_recording.py"), "--dir", str(raw), "--plot"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "[PASS] duration" in out.stdout
    assert (raw / name).with_suffix(".png").stat().st_size > 10_000
