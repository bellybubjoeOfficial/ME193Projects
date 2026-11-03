"""Stage 1 -- session logging round-trips (parquet and csv)."""

import json

import pandas as pd
import pytest

from laptop.acquire.logger import SessionLogger
from laptop.acquire.messages import parse_imu_batch
from laptop.acquire.seqtrack import Gap
from laptop.data.load import load_imu
from test_messages import batch


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_round_trip(tmp_path, fmt):
    lg = SessionLogger(tmp_path, "unoq-shin", "abc-123-session", fmt=fmt, flush_rows=7)
    b = parse_imu_batch(batch())
    lg.log_batch(b, t_arrival_ns=111, offset_ns=None)    # before first sync
    lg.log_batch(b, t_arrival_ns=222, offset_ns=1000)
    lg.close(gaps=[Gap(10, 5, 333, recovered=2)])

    folder = lg.folder
    df = load_imu(folder, sort=False)
    assert len(df) == 10 and lg.rows_written == 10
    assert list(df.columns[:6]) == ["seq", "t_mono_ns", "t_wall_ns", "t_arrival_ns",
                                    "offset_ns", "t_laptop_ns"]
    assert int(df["t_mono_ns"].iloc[0]) == int(b.t_mono_ns[0])  # full ns precision survives
    assert df["t_laptop_ns"].iloc[:5].isna().all()
    assert int(df["t_laptop_ns"].iloc[5]) == int(b.t_mono_ns[0]) - 1000
    assert df["az"].iloc[0] == pytest.approx(-0.2)

    gaps = pd.read_csv(folder / "gaps.csv")
    assert gaps.iloc[0].tolist() == [333, 10, 5, 2]
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["session"] == "abc-123-session"
