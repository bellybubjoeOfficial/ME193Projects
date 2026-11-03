"""Stage 1: MQTT acquisition of the shin IMU.

ImuAcquirer subscribes to the raw/status/pong topics, parses each batch,
tags it with laptop arrival time, tracks seq gaps, converts device time to
laptop time with the running clock offset, fills the ring buffer for the
UI, and logs everything to one folder per device session. A background
thread pings every sync.ping_interval_s.

Run headless (prints stats every 2 s):
    python -m laptop.acquire.acquirer
"""

import json
import threading
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt

from laptop.acquire.logger import SessionLogger
from laptop.acquire.messages import ParseError, parse_imu_batch, parse_pong, parse_status
from laptop.acquire.ringbuffer import RingBuffer
from laptop.acquire.seqtrack import SeqTracker
from laptop.clock import now_ns
from laptop.config import load_config, resolve
from laptop.sync.clocksync import ClockSync


class ImuAcquirer:
    def __init__(self, cfg=None, log=True, out_dir=None):
        """out_dir: log into this folder (Stage 4 recording) instead of data/sessions/."""
        self.cfg = cfg or load_config()
        self.out_dir = out_dir
        m, imu, s = self.cfg["mqtt"], self.cfg["imu"], self.cfg["sync"]
        self.topics = m["topics"]
        self.device_clock = s["device_clock"]  # sample clock that pongs share
        self.log_enabled = log

        self.ring = RingBuffer(int(imu["ring_seconds"] * imu["expected_rate_hz"] * 1.5))
        self.sync = ClockSync(s["median_window"], s["max_rtt_ms"])
        self.seq = SeqTracker()
        self.logger = None
        self.session = None
        self.device = None
        self.rate_hz = float(imu["expected_rate_hz"])
        self.status = {}
        self.status_t_ns = None
        self.bad_messages = 0
        self.batches = 0
        self.sessions_seen = 0
        self.connected = False
        self.disconnects = 0
        self.t_start_ns = None
        self.first_arrival_ns = None
        self.last_arrival_ns = None
        self.first_seq = None
        self.last_seq = None
        self._latency = deque(maxlen=imu["latency_window"])  # arrival - source, ns
        self._arrivals = deque()  # (t_arrival_ns, n_samples) for the rate window
        self._lock = threading.Lock()
        self._ping_id = 0
        self._stop = threading.Event()

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                   client_id=f"laptop-acq-{int(time.time())}")
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=5)

    # ---- lifecycle ----
    def start(self):
        m = self.cfg["mqtt"]
        self.t_start_ns = now_ns()
        self._client.connect_async(m["host"], m["port"], keepalive=m["keepalive_s"])
        self._client.loop_start()
        threading.Thread(target=self._ping_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        self._client.loop_stop()
        self._client.disconnect()
        with self._lock:
            self._close_logger()

    def send_cmd(self, cmd, value=None):
        msg = {"cmd": cmd} if value is None else {"cmd": cmd, "value": value}
        self._client.publish(self.topics["cmd"], json.dumps(msg), qos=1)

    # ---- MQTT callbacks (paho network thread) ----
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            return
        self.connected = True
        for key in ("raw", "status", "pong"):
            client.subscribe(self.topics[key], qos=0)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        if self.connected:
            self.disconnects += 1
        self.connected = False

    def _on_message(self, client, userdata, msg):
        t_arrival = now_ns()
        try:
            if msg.topic == self.topics["raw"]:
                self._handle_batch(parse_imu_batch(msg.payload), t_arrival)
            elif msg.topic == self.topics["pong"]:
                self._handle_pong(parse_pong(msg.payload), t_arrival)
            elif msg.topic == self.topics["status"]:
                status = parse_status(msg.payload)
                with self._lock:
                    self.status, self.status_t_ns = status, t_arrival
                    if self.logger:
                        self.logger.log_status(status, t_arrival)
        except ParseError as e:
            with self._lock:
                self.bad_messages += 1
            print(f"[acquire] bad message on {msg.topic}: {e}")

    def _handle_batch(self, batch, t_arrival):
        with self._lock:
            if batch.session != self.session:
                self._new_session(batch)
            new_gaps = self.seq.update(batch.seq, t_arrival)
            self.batches += 1
            self.rate_hz = batch.rate_hz
            # Seq of each batch's last sample pairs with that batch's arrival time.
            if self.first_arrival_ns is None:
                self.first_arrival_ns, self.first_seq = t_arrival, int(batch.seq.max())
            self.last_arrival_ns = t_arrival
            self.last_seq = max(self.last_seq or 0, int(batch.seq.max()))
            self._arrivals.append((t_arrival, len(batch)))

            offset = self.sync.offset_ns
            if offset is not None:
                t_laptop = self.sync.device_to_laptop(batch.device_time(self.device_clock))
                self._latency.append(t_arrival - int(t_laptop[-1]))
            else:
                # Not synced yet: place samples by arrival time so plots still move.
                period = int(1e9 / batch.rate_hz)
                t_laptop = t_arrival - period * np.arange(len(batch) - 1, -1, -1)
            self.ring.append(t_laptop, batch.seq, batch.data)
            if self.logger:
                self.logger.log_batch(batch, t_arrival, offset, self.device_clock)
        for g in new_gaps:
            print(f"[acquire] GAP: {g.count} samples missing starting at seq {g.first_missing}")

    def _handle_pong(self, pong, t3):
        with self._lock:
            sample = self.sync.add(pong.t_laptop_ns, pong.t_dev_recv_ns, pong.t_dev_send_ns, t3)
            if self.logger:
                self.logger.log_sync(pong, t3, sample, self.sync.offset_ns)

    def _new_session(self, batch):
        """Called with the lock held when the device's session uuid changes."""
        self._close_logger()
        if self.session is not None:
            print(f"[acquire] new device session {batch.session} (was {self.session})")
        self.session, self.device = batch.session, batch.device
        self.sessions_seen += 1
        self.seq = SeqTracker()
        self.ring.clear()
        self._latency.clear()
        self._arrivals.clear()
        self.first_arrival_ns = self.first_seq = self.last_seq = None
        if self.log_enabled:
            lg = self.cfg["logging"]
            folder = None
            if self.out_dir:  # a device reboot mid-recording goes to a side folder
                folder = self.out_dir if self.sessions_seen == 1 else \
                    self.out_dir / f"imu_device_session{self.sessions_seen}"
            self.logger = SessionLogger(resolve(lg["data_dir"]), batch.device, batch.session,
                                        lg["format"], lg["flush_rows"],
                                        meta={"rate_hz": batch.rate_hz, "config": self.cfg},
                                        folder=folder, meta_name="imu_meta.json" if folder else "meta.json")
            print(f"[acquire] logging to {self.logger.folder}")

    def _close_logger(self):
        if self.logger:
            self.logger.close(self.seq.gaps)
            print(f"[acquire] saved {self.logger.rows_written} samples to {self.logger.folder}")
            self.logger = None

    def _ping_loop(self):
        interval = self.cfg["sync"]["ping_interval_s"]
        while not self._stop.wait(interval):
            if self.connected:
                self._ping_id += 1
                self._client.publish(self.topics["ping"], json.dumps(
                    {"id": self._ping_id, "t_laptop_ns": now_ns()}), qos=0)

    # ---- stats for the UI / pass-fail ----
    def stats(self):
        now = now_ns()
        window_ns = int(self.cfg["imu"]["rate_window_s"] * 1e9)
        with self._lock:
            while self._arrivals and self._arrivals[0][0] < now - window_ns:
                self._arrivals.popleft()
            # Samples after the first batch in the window, over the time they took.
            arr = self._arrivals
            window_s = (arr[-1][0] - arr[0][0]) / 1e9 if len(arr) >= 2 else 0.0
            recent = sum(n for _, n in arr) - arr[0][1] if arr else 0
            # Whole-run rate: seq span over laptop-time span (immune to device clock error).
            span_s = ((self.last_arrival_ns or 0) - (self.first_arrival_ns or 0)) / 1e9
            run_rate = (self.last_seq - self.first_seq) / span_s if span_s > 1 else None
            lat = np.array(self._latency) / 1e6 if self._latency else None
            gaps = list(self.seq.gaps)
            return {
                "connected": self.connected,
                "disconnects": self.disconnects,
                "device": self.device,
                "session": self.session,
                "sessions_seen": self.sessions_seen,
                "nominal_rate_hz": self.rate_hz,
                "rate_hz_window": recent / window_s if window_s > 0.5 else None,
                "rate_hz_run": run_rate,
                "received": self.seq.received,
                "expected": self.seq.expected,
                "received_frac": self.seq.received_frac,
                "missing": len(self.seq.missing),
                "late": self.seq.late,
                "duplicates": self.seq.duplicates,
                "gaps": gaps,
                "bad_messages": self.bad_messages,
                "batches": self.batches,
                "latency_ms_median": float(np.median(lat)) if lat is not None else None,
                "latency_ms_p95": float(np.percentile(lat, 95)) if lat is not None else None,
                "offset_ns": self.sync.offset_ns,
                "rtt_ms": self.sync.rtt_ns / 1e6 if self.sync.rtt_ns is not None else None,
                "pongs": len(self.sync.history),
                "drift_ppm": self.sync.drift_ppm(),
                "status": dict(self.status),
                "status_age_s": (now - self.status_t_ns) / 1e9 if self.status_t_ns else None,
                "run_s": (now - self.t_start_ns) / 1e9 if self.t_start_ns else 0.0,
                "log_folder": str(self.logger.folder) if self.logger else None,
            }


def format_stats(s):
    def f(v, fmt):
        return "--" if v is None else format(v, fmt)
    st = s["status"]
    lines = [
        f"MQTT     {'connected' if s['connected'] else 'DISCONNECTED'}"
        f"   broker drops {s['disconnects']}",
        f"device   {s['device'] or '--'}  session {(s['session'] or '--')[:8]}",
        f"rate     {f(s['rate_hz_window'], '.2f')} Hz (last window)   "
        f"{f(s['rate_hz_run'], '.3f')} Hz (whole run)   nominal {s['nominal_rate_hz']:g}",
        f"samples  {s['received']} / {s['expected']} expected   "
        f"({100 * s['received_frac']:.3f}%)",
        f"seq      missing {s['missing']}   late {s['late']}   dup {s['duplicates']}"
        f"   gaps {len(s['gaps'])}   bad msgs {s['bad_messages']}",
        f"latency  median {f(s['latency_ms_median'], '.1f')} ms   "
        f"p95 {f(s['latency_ms_p95'], '.1f')} ms   (arrival - source time)",
        f"clock    offset {f(None if s['offset_ns'] is None else s['offset_ns'] / 1e9, '.6f')} s"
        f"   rtt {f(s['rtt_ms'], '.1f')} ms   drift {f(s['drift_ppm'], '+.1f')} ppm"
        f"   pongs {s['pongs']}",
        f"status   online={st.get('online')}  dev rate {st.get('rate_hz_measured')}  "
        f"sent {st.get('samples_sent')}  buffered {st.get('samples_buffered')}  "
        f"reconnects {st.get('reconnects')}  "
        f"±{st.get('accel_range_g')} g  ±{st.get('gyro_range_dps')} dps  "
        f"up {st.get('uptime_s')} s  (age {f(s['status_age_s'], '.1f')} s)",
    ]
    return "\n".join(lines)


def main():
    acq = ImuAcquirer()
    acq.start()
    print("listening (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(2)
            print(format_stats(acq.stats()), end="\n\n")
    except KeyboardInterrupt:
        pass
    finally:
        acq.stop()


if __name__ == "__main__":
    main()
