"""Load a logged session folder back into pandas.

Always use this instead of pd.read_csv directly: CSV columns with blanks
(t_laptop_ns before the first sync) would otherwise become float64 and lose
nanosecond precision.
"""

from pathlib import Path

import pandas as pd

NS_COLUMNS = ["seq", "t_mono_ns", "t_wall_ns", "t_arrival_ns", "offset_ns", "t_laptop_ns"]


def load_imu(folder, sort=True):
    folder = Path(folder)
    if (folder / "imu.parquet").exists():
        df = pd.read_parquet(folder / "imu.parquet")
    else:
        df = pd.read_csv(folder / "imu.csv", dtype={c: "Int64" for c in NS_COLUMNS})
    if sort:  # logged in arrival order; late samples go back in place
        df = df.sort_values("seq", kind="stable").reset_index(drop=True)
    return df


def load_sync(folder):
    return pd.read_csv(Path(folder) / "sync_log.csv", dtype="Int64")
