"""Write one device session to its own folder.

    <data_dir>/<YYYYmmdd-HHMMSS>_<session[:8]>/
        imu.parquet (or imu.csv)  every sample as it arrived (sort by seq offline)
        sync_log.csv              every ping/pong and the running median offset
        status.jsonl              every device status message
        gaps.csv                  seq gaps (written on close, with recovered counts)
        meta.json                 device, session, start time, config used

Raw device timestamps are always kept, so laptop-time conversion can be
redone offline from sync_log.csv with a better offset.
"""

import csv
import json
import threading
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from laptop.acquire.messages import FIELDS

SYNC_COLUMNS = ["id", "t0_laptop_ns", "t1_dev_recv_ns", "t2_dev_send_ns", "t3_laptop_ns",
                "offset_ns", "rtt_ns", "accepted", "median_offset_ns"]


class SessionLogger:
    def __init__(self, data_dir, device, session, fmt="parquet", flush_rows=1000, meta=None,
                 folder=None, meta_name="meta.json"):
        """folder: write into this exact folder (Stage 4 session) instead of a new one."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.folder = Path(folder) if folder else Path(data_dir) / f"{stamp}_{session[:8]}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt
        self.flush_rows = flush_rows
        self.rows_written = 0
        self._pending = []
        self._pending_rows = 0
        self._writer = None
        self._lock = threading.Lock()

        self.imu_path = self.folder / f"imu.{'parquet' if fmt == 'parquet' else 'csv'}"
        self._sync_file = open(self.folder / "sync_log.csv", "w", newline="")
        self._sync_csv = csv.writer(self._sync_file)
        self._sync_csv.writerow(SYNC_COLUMNS)
        self._status_file = open(self.folder / "status.jsonl", "w")
        with open(self.folder / meta_name, "w") as f:
            json.dump({"device": device, "session": session,
                       "started_wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       **(meta or {})}, f, indent=2)

    def log_batch(self, batch, t_arrival_ns, offset_ns, device_clock="mono"):
        """offset_ns is the median offset at arrival (None before first sync);
        device_clock says which sample clock it converts (see config sync)."""
        n = len(batch)
        df = pd.DataFrame({
            "seq": batch.seq,
            "t_mono_ns": batch.t_mono_ns,
            "t_wall_ns": batch.t_wall_ns,
            "t_arrival_ns": pd.array([t_arrival_ns] * n, dtype="Int64"),
            "offset_ns": pd.array([offset_ns] * n, dtype="Int64"),
            "t_laptop_ns": pd.array(batch.device_time(device_clock) - offset_ns if offset_ns is not None
                                    else [None] * n, dtype="Int64"),
        })
        for i, name in enumerate(FIELDS):
            df[name] = batch.data[:, i]
        with self._lock:
            self._pending.append(df)
            self._pending_rows += n
            if self._pending_rows >= self.flush_rows:
                self._flush()

    def log_sync(self, pong, t3, sample, median_offset_ns):
        with self._lock:
            self._sync_csv.writerow([pong.id, pong.t_laptop_ns, pong.t_dev_recv_ns,
                                     pong.t_dev_send_ns, t3, sample.offset_ns, sample.rtt_ns,
                                     int(sample.accepted), median_offset_ns])
            self._sync_file.flush()

    def log_status(self, status, t_arrival_ns):
        with self._lock:
            self._status_file.write(json.dumps({"t_arrival_ns": t_arrival_ns, **status}) + "\n")
            self._status_file.flush()

    def _flush(self):
        if not self._pending:
            return
        df = pd.concat(self._pending, ignore_index=True)
        self._pending, self._pending_rows = [], 0
        if self.fmt == "parquet":
            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._writer is None:
                self._writer = pq.ParquetWriter(self.imu_path, table.schema)
            self._writer.write_table(table)
        else:
            df.to_csv(self.imu_path, mode="a", header=self.rows_written == 0, index=False)
        self.rows_written += len(df)

    def close(self, gaps=()):
        with self._lock:
            self._flush()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._sync_file.close()
            self._status_file.close()
            with open(self.folder / "gaps.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_detected_ns", "first_missing", "count", "recovered"])
                for g in gaps:
                    w.writerow([g.t_detected_ns, g.first_missing, g.count, g.recovered])
