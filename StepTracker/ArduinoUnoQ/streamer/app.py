"""The streamer: MCU samples in (arduino-router), MQTT out.

Threads:
  router     receives imu_batch notifications, stamps + queues samples
  paho       MQTT network loop (auto-reconnect), ping -> pong, commands in
  publisher  ring buffer -> 10-sample QoS 1 messages, tracks PUBACKs
  status     wearable/status every second (retained), wearable/diag every 5 s
  main       commands, clock sync, MCU state reconcile, LED
"""

import collections
import json
import logging
import queue
import threading
import time
import uuid

import paho.mqtt.client as mqtt

from . import imu
from .clock import ClockMap
from .config import VALID_RATES
from .ring import SampleRing
from .router import RouterClient

log = logging.getLogger("streamer")

FIELDS = ["ax", "ay", "az", "gx", "gy", "gz"]
UNITS = {"accel": "m/s^2", "gyro": "deg/s"}
FW_PREFIX = "shin_imu/"

# LED4 colours, must match set_led() in the sketch.
LED_OFF, LED_GREEN, LED_BLUE, LED_RED = range(4)

SYNC_PERIOD_S = 1.0
SYNC_PERIOD_UNSYNCED_S = 0.1
RECONCILE_PERIOD_S = 2.0
STATUS_PERIOD_S = 1.0
DIAG_PERIOD_S = 5.0
RATE_WINDOW_NS = 5_000_000_000
STOP_DRAIN_S = 0.5  # after stream(false), samples still in flight from the MCU


def parse_stats(text):
    out = {}
    for part in text.split(";"):
        k, _, v = part.partition("=")
        out[k] = v
    return out


def cfg_int(v):
    try:
        return int(v, 0)
    except (TypeError, ValueError):
        return None


class Streamer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t_start = time.monotonic()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # session/key/stat fields shared with the router thread

        self.clock = ClockMap()
        # Sized for the highest rate, so it holds >= buffer_seconds at any rate.
        self.ring = SampleRing(int(cfg.buffer_seconds * max(VALID_RATES)))
        self.pending = collections.deque()  # decoded, waiting for the first clock sync
        self.conv = imu.Converter()

        # Session state
        self.want_streaming = False
        self.rate_hz = cfg.rate_hz
        self.session = None
        self.key = None  # (session, rate_hz) of samples being produced now
        self.seqmap = None
        self.close_session_at = None

        # Stats
        self.rate_window = collections.deque()
        self.samples_in = 0
        self.samples_sent = 0
        self.discarded_stopped = 0
        self.last_t_mcu = None
        self.peak_accel = 0.0
        self.peak_gyro = 0.0
        self.saturated = 0
        self.mcu = {}
        self.fw_ok = None
        self.led = None

        # MQTT
        self.mqtt_connected = threading.Event()
        self.connects = 0
        self.connect_failing = False
        self.inflight = {}  # mid -> n samples, owned by the publisher thread
        self.inflight_samples = 0  # sum of inflight values, readable from any thread
        self.acks = queue.SimpleQueue()
        self.cmds = queue.SimpleQueue()
        self.wake_publisher = threading.Event()

        self.router = RouterClient(cfg.router_socket, on_connect=self._on_router_connect)
        self.router.provide("imu_batch", self.on_batch)
        self.need_reconcile = threading.Event()
        self.client = self._make_client()

    # ------------------------------------------------------------------ MQTT
    def _status_payload(self, online):
        with self.lock:
            buffered = len(self.ring) + len(self.pending) + self.inflight_samples
            return {
                "device": self.cfg.device,
                "online": online,
                "session": self.session,
                "rate_hz_measured": self._measured_rate(),
                "samples_sent": self.samples_sent,
                "samples_buffered": buffered,
                "reconnects": max(0, self.connects - 1),
                "accel_range_g": self.conv.accel_range_g,
                "gyro_range_dps": self.conv.gyro_range_dps,
                "uptime_s": round(time.monotonic() - self.t_start, 1),
            }

    def _make_client(self):
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.cfg.client_id,
                        clean_session=True, protocol=mqtt.MQTTv311)
        if self.cfg.username:
            c.username_pw_set(self.cfg.username, self.cfg.password)
        will = {k: None for k in self._status_payload(False)}
        will.update(device=self.cfg.device, online=False, rate_hz_measured=0)
        c.will_set(self.cfg.topic("status"), json.dumps(will), qos=1, retain=True)
        # Our own window (max_inflight) bounds data messages; leave paho room
        # on top for status / acks so they never wait behind a flush.
        c.max_inflight_messages_set(self.cfg.max_inflight + 10)
        c.max_queued_messages_set(0)
        c.reconnect_delay_set(min_delay=1, max_delay=8)
        c.on_connect = self._on_connect
        c.on_disconnect = self._on_disconnect
        c.on_connect_fail = self._on_connect_fail
        c.on_message = self._on_message
        c.on_publish = lambda client, ud, mid, rc, props: self.acks.put(mid)
        return c

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connect refused: %s", reason_code)
            return
        self.connects += 1
        self.connect_failing = False
        client.subscribe([(self.cfg.topic("sync/ping"), 0), (self.cfg.topic("cmd"), 1)])
        self.mqtt_connected.set()
        self.wake_publisher.set()
        log.info("MQTT connected to %s:%d (connect #%d)", self.cfg.broker_host, self.cfg.broker_port,
                 self.connects)

    def _on_connect_fail(self, client, userdata):
        if not self.connect_failing:  # once per outage, not every retry
            self.connect_failing = True
            log.warning("cannot reach MQTT broker %s:%d; buffering and retrying", self.cfg.broker_host,
                        self.cfg.broker_port)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.mqtt_connected.clear()
        log.warning("MQTT disconnected (%s); buffering", reason_code)

    def _on_message(self, client, userdata, msg):
        if msg.topic == self.cfg.topic("sync/ping"):
            t_recv = time.time_ns()  # first thing: closest to arrival
            try:
                ping = json.loads(msg.payload)
                pong = {"id": ping["id"], "t_laptop_ns": ping["t_laptop_ns"], "t_dev_recv_ns": t_recv}
            except (ValueError, KeyError, TypeError) as e:
                log.warning("bad ping %r: %s", msg.payload[:100], e)
                return
            pong["t_dev_send_ns"] = time.time_ns()
            # QoS 0: a QoS 1 pong could queue behind a backlog flush, and then
            # t_dev_send_ns would no longer be the moment it left.
            client.publish(self.cfg.topic("sync/pong"), json.dumps(pong), qos=0)
        elif msg.topic == self.cfg.topic("cmd"):
            self.cmds.put(msg.payload)

    # ------------------------------------------------------------ MCU input
    def _on_router_connect(self):
        self.need_reconcile.set()

    def on_batch(self, blob):
        """Router thread: one imu_batch notification from the MCU."""
        recs = imu.decode_batch(blob)
        with self.lock:
            self.pending.extend(recs)
            self._drain_pending()

    def _drain_pending(self):
        if not self.clock.ready:
            while len(self.pending) > self.ring.capacity:
                self.pending.popleft()
                self.ring.dropped += 1
            return
        wall_minus_mono = time.time_ns() - time.monotonic_ns()
        period = 1_000_000_000 // self.rate_hz
        out = []
        while self.pending:
            mcu_seq, t_mcu, raw = self.pending.popleft()
            if self.last_t_mcu is not None and t_mcu < self.last_t_mcu:
                log.warning("MCU clock went backwards (MCU reset?): resyncing clock")
                self.clock.reset()
                self.need_reconcile.set()
                self.pending.appendleft((mcu_seq, t_mcu, raw))
                self.last_t_mcu = None
                break
            self.last_t_mcu = t_mcu
            self.samples_in += 1
            if self.key is None:
                self.discarded_stopped += 1
                continue
            t_mono = self.clock.map_sample(t_mcu)
            seq = self.seqmap(mcu_seq, t_mono, period)
            vals = self.conv(raw)
            row = json.dumps([seq, t_mono, t_mono + wall_minus_mono, *vals], separators=(",", ":"))
            out.append((self.key, row))
            self.rate_window.append(t_mono)
            if imu.saturated(raw):
                self.saturated += 1
            self.peak_accel = max(self.peak_accel, *(abs(v) for v in vals[:3]))
            self.peak_gyro = max(self.peak_gyro, *(abs(v) for v in vals[3:]))
        if out:
            while self.rate_window and self.rate_window[-1] - self.rate_window[0] > RATE_WINDOW_NS:
                self.rate_window.popleft()
            self.ring.extend(out)
            self.wake_publisher.set()

    def _measured_rate(self):
        w = self.rate_window
        if len(w) < 2 or time.monotonic_ns() - w[-1] > 1_500_000_000:
            return 0.0
        return round((len(w) - 1) * 1e9 / (w[-1] - w[0]), 2)

    # ------------------------------------------------------------ publisher
    def _raw_message(self, key, rows):
        session, rate = key
        head = json.dumps({"device": self.cfg.device, "session": session, "rate_hz": rate,
                           "fields": FIELDS, "units": UNITS})
        return head[:-1] + ', "samples": [' + ",".join(rows) + "]}"

    def _process_acks(self):
        while True:
            try:
                mid = self.acks.get_nowait()
            except queue.Empty:
                return
            n = self.inflight.pop(mid, None)
            if n is not None:
                with self.lock:
                    self.samples_sent += n
                    self.inflight_samples -= n

    def _track(self, mid, n):
        self.inflight[mid] = n
        with self.lock:
            self.inflight_samples += n

    def publisher_loop(self):
        topic = self.cfg.topic("imu/raw")
        while not self.stop_event.is_set():
            self.wake_publisher.wait(0.05)
            self.wake_publisher.clear()
            self._process_acks()
            # Only hand data to paho while connected and the window has room:
            # unacked messages stay in paho (re-sent on reconnect, in order),
            # everything else waits in our ring buffer.
            while (self.mqtt_connected.is_set() and len(self.inflight) < self.cfg.max_inflight
                   and not self.stop_event.is_set()):
                batch = self.ring.take(self.cfg.batch_size, self.key)
                if batch is None:
                    break
                key, rows = batch
                info = self.client.publish(topic, self._raw_message(key, rows), qos=self.cfg.qos)
                if info.rc not in (mqtt.MQTT_ERR_SUCCESS, mqtt.MQTT_ERR_NO_CONN):
                    # NO_CONN is fine: paho keeps QoS 1 messages and sends them
                    # after reconnecting. Anything else: retry later.
                    log.warning("publish refused (%s); will retry", mqtt.error_string(info.rc))
                    self.ring.push_front(key, rows)
                    break
                self._track(info.mid, len(rows))
                self._process_acks()

    # --------------------------------------------------------------- status
    def _diag_payload(self):
        with self.lock:
            return {
                "device": self.cfg.device,
                "session": self.session,
                "rate_hz_cfg": self.rate_hz,
                "streaming": self.want_streaming,
                "source": self.mcu.get("src"),
                "fw": self.mcu.get("fw"),
                "mcu": {k: self.mcu.get(k) for k in ("bus", "addr", "who", "accel_cfg", "gyro_cfg", "dlpf",
                                                     "rate", "late", "i2c_err", "overflow", "sent")},
                "seq_gaps": self.seqmap.gaps if self.seqmap else 0,
                "ring_dropped": self.ring.dropped,
                "discarded_while_stopped": self.discarded_stopped,
                "peak_accel_ms2": self.peak_accel,
                "peak_gyro_dps": self.peak_gyro,
                "saturated_samples": self.saturated,
                "clock": self.clock.info(),
                "mqtt_inflight": len(self.inflight),
            }

    def status_loop(self):
        next_diag = 0.0
        while not self.stop_event.wait(STATUS_PERIOD_S - (time.monotonic() - self.t_start) % STATUS_PERIOD_S):
            if not self.mqtt_connected.is_set():
                continue  # don't queue stale statuses for later
            self.client.publish(self.cfg.topic("status"), json.dumps(self._status_payload(True)),
                                qos=1, retain=True)
            if time.monotonic() >= next_diag:
                next_diag = time.monotonic() + DIAG_PERIOD_S
                self.client.publish(self.cfg.topic("diag"), json.dumps(self._diag_payload()), qos=0, retain=True)

    # ------------------------------------------------------- control (main)
    def start_session(self):
        with self.lock:
            self.session = str(uuid.uuid4())
            self.seqmap = imu.SessionSeq()
            self.key = (self.session, self.rate_hz)
            self.want_streaming = True
            self.close_session_at = None
        log.info("session %s started at %d Hz", self.session, self.rate_hz)
        self._mcu_call("stream", True)

    def stop_session(self):
        self.want_streaming = False
        self._mcu_call("stream", False)
        self.close_session_at = time.monotonic() + STOP_DRAIN_S

    def _close_session(self):
        with self.lock:
            log.info("session %s stopped (gaps: %d)", self.session, self.seqmap.gaps if self.seqmap else 0)
            self.session = self.key = None
            self.close_session_at = None
        self.wake_publisher.set()  # flush the last partial batch

    def handle_cmd(self, payload):
        try:
            msg = json.loads(payload)
            cmd = msg["cmd"]
        except (ValueError, KeyError, TypeError):
            return self._ack(None, False, f"bad command: {payload[:200]!r}")
        if cmd == "start":
            if self.want_streaming:
                return self._ack(cmd, True, f"already streaming, session {self.session}")
            if self.close_session_at is not None:
                self._close_session()
            self.start_session()
            return self._ack(cmd, True, f"session {self.session}")
        if cmd == "stop":
            if not self.want_streaming:
                return self._ack(cmd, True, "already stopped")
            self.stop_session()
            return self._ack(cmd, True, f"stopping session {self.session}")
        if cmd == "set_rate":
            try:
                hz = int(msg.get("value"))
            except (TypeError, ValueError):
                hz = None
            if hz not in VALID_RATES:
                return self._ack(cmd, False, f"value must be one of {list(VALID_RATES)}")
            if not self._mcu_call("set_rate", hz):
                return self._ack(cmd, False, "MCU refused or did not answer")
            with self.lock:
                self.rate_hz = hz
                self.rate_window.clear()
                if self.session:
                    self.key = (self.session, hz)
            log.info("rate set to %d Hz", hz)
            return self._ack(cmd, True, f"rate {hz} Hz")
        return self._ack(cmd, False, f"unknown cmd {cmd!r}")

    def _ack(self, cmd, ok, detail):
        (log.info if ok else log.warning)("cmd %s: %s", cmd, detail)
        self.client.publish(self.cfg.topic("cmd/ack"), json.dumps({"cmd": cmd, "ok": ok, "detail": detail}), qos=1)

    def _mcu_call(self, method, *args, timeout=2.0):
        try:
            return self.router.call(method, *args, timeout=timeout)
        except Exception as e:  # noqa: BLE001 - logged; callers check the result
            log.warning("MCU %s%r failed: %s", method, args, e)
            return None

    def sync_clock(self):
        t0 = time.monotonic_ns()
        t_mcu = self._mcu_call("now_ns", timeout=1.0)
        t1 = time.monotonic_ns()
        if t_mcu is None:
            return
        if not self.clock.add(t0, t_mcu, t1):
            log.warning("MCU clock restarted: resetting clock map")
            self.clock.reset()
            self.clock.add(t0, t_mcu, t1)
        if self.pending:
            with self.lock:
                self._drain_pending()

    def reconcile(self):
        text = self._mcu_call("mcu_stats")
        if text is None:
            return
        st = parse_stats(text)
        fw_ok = st.get("fw", "").startswith(FW_PREFIX)
        if fw_ok != self.fw_ok:
            self.fw_ok = fw_ok
            if not fw_ok:
                log.error("the MCU is not running the shin_imu sketch (got %r); run scripts/flash_sketch.sh",
                          text[:80])
        if not fw_ok:
            return
        imu_ok = st.get("ok") == "1"
        if imu_ok:
            a, g = cfg_int(st.get("accel_cfg")), cfg_int(st.get("gyro_cfg"))
            if a is not None and g is not None:
                rng = imu.ranges_from_cfg(a, g)
                if rng != (self.conv.accel_range_g, self.conv.gyro_range_dps):
                    self.conv = imu.Converter(*rng)
        if st.get("src") != self.mcu.get("src"):
            log.info("MCU: %s", text)
        self.mcu = st
        if not imu_ok and self.cfg.synthetic_if_no_imu and st.get("src") != "synthetic":
            log.warning("no IMU found: enabling the MCU's synthetic test pattern (config synthetic_if_no_imu)")
            self._mcu_call("set_synthetic", True)
        if str(self.rate_hz) != st.get("rate"):
            self._mcu_call("set_rate", self.rate_hz)
        if ("1" if self.want_streaming else "0") != st.get("streaming"):
            self._mcu_call("stream", self.want_streaming)

    def update_led(self):
        src = self.mcu.get("src")
        if not self.want_streaming:
            led = LED_OFF
        elif src not in ("imu", "synthetic"):
            led = LED_RED
        else:
            led = LED_GREEN if self.mqtt_connected.is_set() else LED_BLUE
        if led != self.led:
            try:
                self.router.notify("set_led", led)
                self.led = led
            except ConnectionError:
                pass

    def log_summary(self):
        st = self._status_payload(self.mqtt_connected.is_set())
        log.info("rate %.2f Hz | sent %d | buffered %d | reconnects %d | gaps %d | dropped %d | clock %s",
                 st["rate_hz_measured"], st["samples_sent"], st["samples_buffered"], st["reconnects"],
                 self.seqmap.gaps if self.seqmap else 0, self.ring.dropped, self.clock.info())

    # ------------------------------------------------------------ lifecycle
    def run(self):
        log.info("streamer starting: broker %s:%d, device %s", self.cfg.broker_host, self.cfg.broker_port,
                 self.cfg.device)
        self.router.start()
        self.client.connect_async(self.cfg.broker_host, self.cfg.broker_port, keepalive=self.cfg.keepalive_s)
        self.client.loop_start()
        self.threads = [threading.Thread(target=self.publisher_loop, name="publisher", daemon=True),
                        threading.Thread(target=self.status_loop, name="status", daemon=True)]
        for t in self.threads:
            t.start()
        if self.cfg.autostart:
            with self.lock:  # session exists from the start; streaming turns on once the MCU answers
                self.session = str(uuid.uuid4())
                self.seqmap = imu.SessionSeq()
                self.key = (self.session, self.rate_hz)
                self.want_streaming = True
            log.info("session %s (autostart, %d Hz)", self.session, self.rate_hz)
        next_sync = next_reconcile = next_summary = 0.0
        while not self.stop_event.wait(0.02):
            now = time.monotonic()
            while True:
                try:
                    self.handle_cmd(self.cmds.get_nowait())
                except queue.Empty:
                    break
            if self.close_session_at is not None and now >= self.close_session_at:
                self._close_session()
            if not self.router.connected:
                continue
            if now >= next_sync:
                self.sync_clock()
                next_sync = now + (SYNC_PERIOD_S if self.clock.ready else SYNC_PERIOD_UNSYNCED_S)
            if now >= next_reconcile or self.need_reconcile.is_set():
                self.need_reconcile.clear()
                self.reconcile()
                next_reconcile = now + RECONCILE_PERIOD_S
            self.update_led()
            if now >= next_summary:
                next_summary = now + 10
                self.log_summary()
        self.shutdown()

    def shutdown(self, flush_timeout_s=3.0):
        log.info("shutting down")
        self.want_streaming = False
        self._mcu_call("stream", False, timeout=1.0)
        try:
            self.router.notify("set_led", LED_OFF)
        except ConnectionError:
            pass
        with self.lock:
            self.session = self.key = None
        for t in self.threads:
            t.join(timeout=2)
        # The publisher thread has stopped: flush what is left from here.
        deadline = time.monotonic() + flush_timeout_s
        topic = self.cfg.topic("imu/raw")
        while self.mqtt_connected.is_set() and time.monotonic() < deadline:
            self._process_acks()
            batch = self.ring.take(self.cfg.batch_size, None)
            if batch is None and not self.inflight:
                break
            if batch is not None:
                info = self.client.publish(topic, self._raw_message(*batch), qos=self.cfg.qos)
                self._track(info.mid, len(batch[1]))
            else:
                time.sleep(0.02)
        left = len(self.ring) + self.inflight_samples
        if left:
            log.error("exiting with %d samples not delivered", left)
        info = self.client.publish(self.cfg.topic("status"), json.dumps(self._status_payload(False)),
                                   qos=1, retain=True)
        if self.mqtt_connected.is_set():
            try:
                info.wait_for_publish(timeout=2)
            except RuntimeError:
                pass
        self.client.disconnect()
        self.client.loop_stop()
        self.router.stop()
