"""Stage 4: load recordings for processing, trimming both ends.

The raw CSVs in data/raw/ are never modified. Trimming happens here, every
time the data is loaded, so you can change TRIM_SECONDS later and re-run.

    recs = dataset.load_recordings()            # [(df, activity, session_id), ...]
    features.build_dataset(recs)                # (Stage 5)
"""

from pathlib import Path

from wearable import config, imu, recording


def trim(df, seconds=config.TRIM_SECONDS, fs=config.SAMPLE_RATE_HZ):
    """Drop `seconds` from the start and the end. Too short -> empty table (same columns)."""
    n = int(round(seconds * fs))
    if len(df) <= 2 * n:
        return df.iloc[0:0].reset_index(drop=True)
    return df.iloc[n:len(df) - n].reset_index(drop=True)


def load_recordings(raw_dir=config.RAW_DIR, sessions_csv=config.SESSIONS_CSV,
                    trim_s=config.TRIM_SECONDS, activities=None):
    """Every finished recording listed in sessions.csv, trimmed.

    Returns a list of (df, activity, session_id) with session_id = file name
    without .csv (unique per recording, used for grouped cross-validation).
    Rows whose file is missing (deleted, or not copied to the laptop) are
    skipped. `activities` limits which ones are loaded (default: all).
    """
    raw_dir = Path(raw_dir)
    out = []
    for row in recording.read_sessions(sessions_csv):
        if activities is not None and row["activity"] not in activities:
            continue
        path = raw_dir / row["file"]
        if not path.exists():
            continue
        out.append((trim(imu.load_csv(path), trim_s), row["activity"], path.stem))
    return out
