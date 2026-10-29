"""Stage 3 hardware test: validates the newest REAL ~60 s sitting recording in data/raw/.

Skipped until one exists (python scripts/record.py sit). Run on the Q, or on
the laptop after scripts/fetch_data.py.
"""

import pandas as pd
import pytest

from wearable import config, imu, validate

pytestmark = pytest.mark.hardware


@pytest.fixture(scope="module")
def sitting():
    # Newest sitting recording of about a minute: the Stage 3 test recording.
    # (Shorter "sit" files from Stage 4 data collection don't count.)
    files = sorted(config.RAW_DIR.glob(f"{config.FILE_PREFIX}_sit_s*.csv"), key=lambda p: p.stat().st_mtime)
    for path in reversed(files):
        df = imu.load_csv(path)
        if len(df) >= 55 * config.SAMPLE_RATE_HZ:
            return path, df
    pytest.skip(f"no ~60 s sitting recording in {config.RAW_DIR} yet — run scripts/record.py sit --seconds 60")


def test_sitting_recording_is_logged(sitting):
    path, _ = sitting
    assert config.SESSIONS_CSV.exists(), "data/sessions.csv missing"
    assert path.name in set(pd.read_csv(config.SESSIONS_CSV)["file"])


@pytest.mark.parametrize("check", ["monotonic", "sample_rate", "values_update", "clipping", "sensible"])
def test_sitting_recording(sitting, check):
    _, df = sitting
    r = {
        "monotonic": validate.check_monotonic,
        "sample_rate": validate.checks.check_sample_rate,
        "values_update": validate.checks.check_values_update,
        "clipping": validate.check_clipping,
        "sensible": lambda d: validate.check_sensible_values(d, "sit"),
    }[check](df)
    assert r.ok, r.detail


def test_sitting_recording_about_a_minute(sitting):
    _, df = sitting
    assert len(df) / config.SAMPLE_RATE_HZ >= 55
