"""Stage 1 hardware test: checks the REAL capture written by the running app.

Skipped until data/check/imu_check.csv exists (i.e. the app has run for
CHECK_SECONDS with the board sitting still). Run on the Q, or on the laptop
after copying data/check/ over.
"""

import json

import pytest

from wearable import checks, config, imu

pytestmark = pytest.mark.hardware

INFO = config.CHECK_DIR / "imu_info.json"
CAPTURE = config.CHECK_DIR / "imu_check.csv"


@pytest.fixture(scope="module")
def info():
    if not INFO.exists():
        pytest.skip(f"no {INFO} yet — run the app first")
    return json.loads(INFO.read_text())


@pytest.fixture(scope="module")
def capture():
    if not CAPTURE.exists():
        pytest.skip(f"no {CAPTURE} yet — let the app record {config.CHECK_SECONDS} s first")
    return imu.load_csv(CAPTURE)


def test_device_found(info):
    r = checks.check_device_found(info)
    assert r.ok, r.detail


def test_who_am_i(info):
    r = checks.check_who_am_i(info)
    assert r.ok, r.detail


def test_ranges_configured(info):
    r = checks.check_ranges(info)
    assert r.ok, r.detail


def test_values_update(capture):
    r = checks.check_values_update(capture)
    assert r.ok, r.detail


def test_accel_1g_when_still(capture):
    r = checks.check_accel_1g(capture)
    assert r.ok, r.detail


def test_gyro_zero_when_still(capture):
    r = checks.check_gyro_still(capture)
    assert r.ok, r.detail


def test_sample_rate_100hz_no_drops(capture):
    r = checks.check_sample_rate(capture)
    assert r.ok, r.detail
