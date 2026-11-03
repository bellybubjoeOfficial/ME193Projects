"""Streamer end to end, no hardware: fake router/MCU -> streamer -> real Mosquitto.

Needs the `mosquitto` broker binary (Debian: /usr/sbin/mosquitto); skipped
otherwise. Takes ~30 s.
"""

import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import pytest

from streamer.app import Streamer
from streamer.config import Config
from tests.fake_router import FakeRouter

MOSQUITTO = shutil.which("mosquitto") or ("/usr/sbin/mosquitto" if Path("/usr/sbin/mosquitto").exists() else None)
pytestmark = pytest.mark.skipif(MOSQUITTO is None, reason="mosquitto broker not installed")

STATUS_KEYS = ["device", "online", "session", "rate_hz_measured", "samples_sent", "samples_buffered",
               "reconnects", "accel_range_g", "gyro_range_dps", "uptime_s"]
RAW_KEYS = ["device", "session", "rate_hz", "fields", "units", "samples"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Broker:
    def __init__(self, tmp, port):
        self.port = port
        self.conf = tmp / "mosquitto.conf"
        (tmp / "persist").mkdir()
        self.conf.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\n"
                             f"persistence true\npersistence_location {tmp}/persist/\n")
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([MOSQUITTO, "-c", str(self.conf)], stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("broker did not start")

    def stop(self):
        self.proc.terminate()
        self.proc.wait(5)


class Collector:
    """Persistent-session subscriber, like tools/record.py."""

    def __init__(self, port):
        self.raw, self.status, self.acks, self.pongs = [], [], [], []
        self.lock = threading.Lock()
        c = self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test-collector", clean_session=False)
        c.on_connect = lambda c, *_: c.subscribe([("wearable/imu/raw", 1), ("wearable/status", 1),
                                                  ("wearable/cmd/ack", 1), ("wearable/sync/pong", 0)])
        c.on_message = self.on_message
        c.reconnect_delay_set(1, 1)
        c.connect(("127.0.0.1"), port, keepalive=5)
        c.loop_start()

    def on_message(self, c, ud, msg):
        p = json.loads(msg.payload)
        with self.lock:
            {"wearable/imu/raw": self.raw, "wearable/status": self.status,
             "wearable/cmd/ack": self.acks, "wearable/sync/pong": self.pongs}[msg.topic].append(p)

    def samples(self, session):
        with self.lock:
            return [r for m in self.raw if m["session"] == session for r in m["samples"]]

    def wait(self, cond, timeout=10):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if cond():
                return True
            time.sleep(0.05)
        return False


def check_consecutive(rows):
    seqs = sorted({r[0] for r in rows})
    assert seqs[0] == 0
    assert seqs == list(range(len(seqs))), f"gaps: {sorted(set(range(seqs[-1] + 1)) - set(seqs))[:10]}"


@pytest.fixture
def rig(tmp_path):
    port = free_port()
    broker = Broker(tmp_path, port)
    broker.start()
    fake = FakeRouter(str(tmp_path / "router.sock")).start()
    col = Collector(port)
    cfg = Config(broker_host="127.0.0.1", broker_port=port, keepalive_s=2,
                 router_socket=str(tmp_path / "router.sock"), buffer_seconds=60)
    s = Streamer(cfg)
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    yield broker, fake, col, s
    s.stop_event.set()
    t.join(10)
    col.c.loop_stop()
    fake.stop()
    broker.stop()


def test_stream_contract_reconnect_sync_and_commands(rig):
    broker, fake, col, s = rig

    # --- streaming + message contract ---------------------------------
    assert col.wait(lambda: len(col.raw) >= 20), "no raw messages"
    m = col.raw[0]
    assert list(m) == RAW_KEYS
    assert m["device"] == "unoq-shin" and m["rate_hz"] == 100
    assert m["fields"] == ["ax", "ay", "az", "gx", "gy", "gz"]
    assert m["units"] == {"accel": "m/s^2", "gyro": "deg/s"}
    assert all(len(x["samples"]) == 10 for x in col.raw)
    row = m["samples"][0]
    assert len(row) == 9 and all(isinstance(v, int) for v in row[:3])
    assert row[5] == pytest.approx(9.807, abs=0.01)  # fake MCU sends az = 2048 counts = 1 g
    session = m["session"]

    # timestamps: MCU clock runs 40 ppm fast with an offset; mapped to Linux time
    rows = col.samples(session)
    d = [b[1] - a[1] for a, b in zip(rows, rows[1:])]
    assert sum(d) / len(d) == pytest.approx(10_000_000, rel=0.01)
    assert abs(rows[-1][1] - time.monotonic_ns()) < 1e9
    assert abs(rows[-1][2] - time.time_ns()) < 1e9

    # --- status ---------------------------------------------------------
    assert col.wait(lambda: len(col.status) >= 2)
    st = col.status[-1]
    assert list(st) == STATUS_KEYS
    assert st["online"] is True and st["session"] == session
    assert st["accel_range_g"] == 16 and st["gyro_range_dps"] == 2000
    assert col.wait(lambda: abs(col.status[-1]["rate_hz_measured"] - 100) < 2), col.status[-1]

    # --- sync ping --------------------------------------------------------
    t1 = time.time_ns()
    col.c.publish("wearable/sync/ping", json.dumps({"id": "p1", "t_laptop_ns": t1}))
    assert col.wait(lambda: col.pongs)
    p = col.pongs[0]
    assert list(p) == ["id", "t_laptop_ns", "t_dev_recv_ns", "t_dev_send_ns"]
    assert p["id"] == "p1" and p["t_laptop_ns"] == t1
    assert t1 <= p["t_dev_recv_ns"] <= p["t_dev_send_ns"] <= time.time_ns()

    # --- broker down 4 s: samples buffer, then all arrive in order ------
    broker.stop()
    time.sleep(4)
    assert s.inflight_samples + len(s.ring) >= 300
    broker.start()
    assert col.wait(lambda: s.mqtt_connected.is_set() and len(s.ring) < 10 and not s.inflight, 20)
    time.sleep(1)
    check_consecutive(col.samples(session))
    assert s.connects >= 2
    assert col.wait(lambda: col.status[-1]["reconnects"] >= 1)

    # --- set_rate ---------------------------------------------------------
    col.c.publish("wearable/cmd", json.dumps({"cmd": "set_rate", "value": 50}), qos=1)
    assert col.wait(lambda: any(a["cmd"] == "set_rate" for a in col.acks))
    assert [a for a in col.acks if a["cmd"] == "set_rate"][-1]["ok"] is True
    assert col.wait(lambda: any(x["rate_hz"] == 50 for x in col.raw[-5:]))
    col.c.publish("wearable/cmd", json.dumps({"cmd": "set_rate", "value": 33}), qos=1)
    assert col.wait(lambda: any(a["cmd"] == "set_rate" and not a["ok"] for a in col.acks))

    # --- stop: stream ends, the final partial batch is flushed ------------
    col.c.publish("wearable/cmd", json.dumps({"cmd": "stop"}), qos=1)
    assert col.wait(lambda: fake.streaming is False)
    assert col.wait(lambda: col.status[-1]["session"] is None, 5)
    n = len(col.raw)
    time.sleep(1.5)
    assert len(col.raw) == n, "messages kept coming after stop"
    check_consecutive(col.samples(session))
    assert s.seqmap.gaps == 0

    # --- start: new session, seq from 0 -------------------------------------
    col.c.publish("wearable/cmd", json.dumps({"cmd": "start"}), qos=1)
    assert col.wait(lambda: col.raw[-1]["session"] != session)
    new = col.raw[-1]["session"]
    assert col.wait(lambda: len(col.samples(new)) >= 50)
    check_consecutive(col.samples(new))


def test_clean_shutdown_publishes_offline(rig):
    broker, fake, col, s = rig
    assert col.wait(lambda: len(col.status) >= 1)
    s.stop_event.set()
    assert col.wait(lambda: col.status[-1]["online"] is False, 10)
    assert fake.streaming is False


def test_last_will_on_crash(tmp_path):
    """Kill the client's TCP link without DISCONNECT: the broker must publish the will."""
    port = free_port()
    broker = Broker(tmp_path, port)
    broker.start()
    try:
        col = Collector(port)
        cfg = Config(broker_host="127.0.0.1", broker_port=port, keepalive_s=1,
                     router_socket=str(tmp_path / "none.sock"))
        s = Streamer(cfg)
        s.client.connect("127.0.0.1", port, keepalive=1)
        s.client.loop_start()
        assert col.wait(lambda: s.mqtt_connected.is_set())
        s.client.loop_stop()          # stop answering: like a crashed process
        s.client.socket().close()
        assert col.wait(lambda: col.status and col.status[-1]["online"] is False, 10)
        assert col.status[-1]["device"] == "unoq-shin"
        col.c.loop_stop()
    finally:
        broker.stop()
