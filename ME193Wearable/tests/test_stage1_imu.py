"""Stage 1 unit tests: no hardware needed (synthetic captures)."""

import numpy as np
import pytest

from conftest import make_still_capture
from wearable import checks, imu
from wearable.checks import FAIL, PASS, WARN


# ---- Unit conversion --------------------------------------------------------

def test_accel_counts_to_g_at_16g():
    assert imu.counts_to_g(2048) == pytest.approx(1.0)
    assert imu.counts_to_g(-32768) == pytest.approx(-16.0)


def test_gyro_counts_to_dps_at_2000():
    assert imu.counts_to_dps(164) == pytest.approx(10.0)
    assert imu.counts_to_dps(32767) == pytest.approx(1998, abs=1)


def test_sample_row_matches_csv_columns():
    row = imu.sample_row(7, 123456, 1.7e9, (0, 0, 2048, 0, 16, -164))
    assert len(row) == len(imu.CSV_COLUMNS)
    assert row[:2] == [7, 123456]
    assert row[5] == pytest.approx(1.0)
    assert row[8] == pytest.approx(-10.0)


# ---- micros() wraparound -----------------------------------------------------

def test_unwrapper_handles_uint32_wrap():
    u = imu.Unwrapper()
    raw = [2**32 - 20000, 2**32 - 10000, 0, 10000]
    out = [u(v) for v in raw]
    assert np.all(np.diff(out) == 10000)


def test_unwrapper_passthrough_without_wrap():
    u = imu.Unwrapper()
    assert [u(v) for v in (0, 10, 20)] == [0, 10, 20]


# ---- Sketch info string ------------------------------------------------------

def test_parse_info_good(good_info):
    assert good_info["ok"] is True
    assert good_info["bus"] == "Wire2"
    assert good_info["addr"] == 0x68
    assert good_info["who_am_i"] == 0x68
    assert good_info["scan"] == {"Wire2": [0x68], "Wire1": []}


def test_parse_info_nothing_found():
    info = imu.parse_info("ok=0;bus=none;addr=none;who=none;accel_cfg=none;gyro_cfg=none;"
                          "scan=Wire2:|Wire1:")
    assert info["ok"] is False and info["addr"] is None
    assert checks.check_device_found(info).status == FAIL


def test_parse_info_rejects_garbage():
    with pytest.raises(ValueError):
        imu.parse_info("hello")


# ---- Info checks ------------------------------------------------------------

def test_device_found_on_a4_a5_bus_at_0x69():
    info = imu.parse_info("ok=1;bus=Wire2;addr=0x69;who=0x68;accel_cfg=0x18;gyro_cfg=0x18;"
                          "scan=Wire2:0x69|Wire1:")
    assert checks.check_device_found(info).status == PASS


def test_wrong_device_only_fails():
    info = imu.parse_info("ok=0;bus=none;addr=none;who=none;accel_cfg=none;gyro_cfg=none;"
                          "scan=Wire2:0x3c|Wire1:")
    r = checks.check_device_found(info)
    assert r.status == FAIL and "0x3c" in r.detail


@pytest.mark.parametrize("who,status", [(0x68, PASS), (0x70, WARN), (0x71, WARN), (0x00, FAIL), (None, FAIL)])
def test_who_am_i(good_info, who, status):
    assert checks.check_who_am_i({**good_info, "who_am_i": who}).status == status


def test_ranges_readback(good_info):
    assert checks.check_ranges(good_info).status == PASS
    assert checks.check_ranges({**good_info, "accel_cfg": 0x00}).status == FAIL


# ---- Capture checks: a good still capture passes everything ----------------

def test_good_still_capture_passes_all(good_info, still_df):
    results = checks.run_all(good_info, still_df)
    assert [r.status for r in results] == [PASS] * len(results), results


def test_tilted_board_still_reads_1g():
    tilt = np.deg2rad(35)
    df = make_still_capture(gravity=(np.sin(tilt), 0, np.cos(tilt)))
    assert checks.check_accel_1g(df).status == PASS


def test_accel_magnitude_off_fails():
    df = make_still_capture(gravity=(0, 0, 1.2))
    assert checks.check_accel_1g(df).status == FAIL


def test_moving_board_warns():
    df = make_still_capture(accel_noise=0.05)
    assert checks.check_accel_1g(df).status == WARN


@pytest.mark.parametrize("bias,status", [((1, 1, 1), PASS), ((8, 0, 0), WARN), ((0, 30, 0), FAIL)])
def test_gyro_bias_levels(bias, status):
    assert checks.check_gyro_still(make_still_capture(gyro_bias=bias)).status == status


def test_frozen_sensor_fails(still_df):
    df = still_df.copy()
    df[imu.ACCEL_COLS + imu.GYRO_COLS] = df.iloc[0][imu.ACCEL_COLS + imu.GYRO_COLS].to_numpy()
    assert checks.check_values_update(df).status == FAIL


# ---- Sample-rate checks -----------------------------------------------------

def test_dropped_samples_detected(still_df):
    df = still_df.drop(index=[500, 501, 502]).reset_index(drop=True)
    r = checks.check_sample_rate(df)
    assert r.status == FAIL and "3 dropped" in r.detail


def test_wrong_rate_detected():
    df = make_still_capture(fs=95)
    assert checks.check_sample_rate(df).status == FAIL


def test_bridge_backlog_detected(still_df):
    df = still_df.copy()
    df["t_host_s"] += np.linspace(0, 2.0, len(df))   # Linux falling 2 s behind
    r = checks.check_sample_rate(df)
    assert r.status == FAIL and "falling behind" in r.detail


def test_good_rate_passes(still_df):
    r = checks.check_sample_rate(still_df)
    assert r.status == PASS and "dropped 0" in r.detail


# ---- CSV round-trip --------------------------------------------------------

def test_csv_roundtrip(tmp_path, still_df):
    path = tmp_path / "cap.csv"
    still_df.to_csv(path, index=False)
    loaded = imu.load_csv(path)
    assert list(loaded.columns) == imu.CSV_COLUMNS
    assert len(loaded) == len(still_df)


def test_csv_wrong_columns_rejected(tmp_path, still_df):
    path = tmp_path / "bad.csv"
    still_df.rename(columns={"ax_g": "ax"}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        imu.load_csv(path)
