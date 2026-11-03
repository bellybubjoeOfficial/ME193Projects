"""
fake_unoq.py -- pretends to be the shin UNO Q, following the README message contract.

Publishes walking-like IMU batches on wearable/imu/raw, status every 1 s,
answers sync pings, and obeys start/stop/set_rate commands. Its "device
clock" is the laptop clock plus an offset and a drift (config.yaml
fake_device), so the clock-sync math has something real to recover.

Fault injection for testing the laptop side:
    --outage-at 60 --outage-for 5          go silent 5 s (samples lost -> a gap)
    --outage-at 60 --outage-for 5 --buffer hold samples, send them on return
    --reorder 0.02                         swap 2% of batches (out-of-order)
    --drop 0.001                           silently drop 0.1% of batches

Usage (from ME193WalkWearable/, with mosquitto running):
    python tests/fake_unoq.py
    python tests/fake_unoq.py --outage-at 30 --outage-for 3 --duration 60
"""

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paho.mqtt.client as mqtt

from laptop.config import load_config


class FakeDevice:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        fd = cfg["fake_device"]
        self.topics = cfg["mqtt"]["topics"]
        self.batch_size = fd["batch_size"]
        self.offset_ns = int(fd["clock_offset_s"] * 1e9)
        self.drift = fd["drift_ppm"] * 1e-6
        self.delay_ms = fd["net_delay_ms"]
        self.rate_hz = float(args.rate or cfg["imu"]["expected_rate_hz"])
        self.session = str(uuid.uuid4())
        self.running = True
        self.sent = 0
        self.reconnects = 0
        self.t0_ns = time.monotonic_ns()
        self.wall_minus_mono = time.time_ns() - self.t0_ns
        self.sync_clock = cfg["sync"]["device_clock"]
        self._held = None  # batch waiting to be swapped (reorder)

        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"fake-unoq-{self.session[:8]}")
        c.will_set(self.topics["status"], json.dumps({"device": "unoq-shin", "online": False}),
                   qos=1, retain=True)
        c.on_connect = lambda cl, *_: cl.subscribe([(self.topics["ping"], 0),
                                                    (self.topics["cmd"], 1)])
        c.on_message = self._on_message
        self.client = c

    def dev_mono_ns(self, laptop_ns=None):
        t = time.monotonic_ns() if laptop_ns is None else laptop_ns
        return int(self.offset_ns + t + (t - self.t0_ns) * self.drift)

    def dev_wall_ns(self, laptop_ns=None):
        """Device wall clock: Unix time, drifting like the mono clock."""
        return self.dev_mono_ns(laptop_ns) - self.offset_ns + self.wall_minus_mono

    def dev_sync_ns(self):
        """The clock the real Q stamps pongs with (config sync.device_clock)."""
        return self.dev_wall_ns() if self.sync_clock == "wall" else self.dev_mono_ns()

    def _on_message(self, client, userdata, msg):
        m = json.loads(msg.payload)
        if msg.topic == self.topics["ping"]:
            threading.Thread(target=self._pong, args=(m,), daemon=True).start()
        elif msg.topic == self.topics["cmd"]:
            print(f"[fake] cmd {m}")
            if m["cmd"] == "start":
                self.running = True
            elif m["cmd"] == "stop":
                self.running = False
            elif m["cmd"] == "set_rate":
                self.rate_hz = float(m["value"])

    def _pong(self, ping):
        time.sleep(random.uniform(*self.delay_ms) / 1000)   # inbound network leg
        t1 = self.dev_sync_ns()
        time.sleep(random.uniform(0.2, 1.0) / 1000)          # device processing
        t2 = self.dev_sync_ns()
        time.sleep(random.uniform(*self.delay_ms) / 1000)   # outbound leg
        self.client.publish(self.topics["pong"], json.dumps(
            {"id": ping["id"], "t_laptop_ns": ping["t_laptop_ns"],
             "t_dev_recv_ns": t1, "t_dev_send_ns": t2}))

    def sample(self, seq, t_laptop_ns):
        """Walking-ish shin signal: ~0.9 Hz stride, impact spike each heel strike."""
        t = (t_laptop_ns - self.t0_ns) / 1e9
        ph = 2 * math.pi * 0.9 * t
        impact = 15 * math.exp(-((t * 0.9) % 1.0) * 40)
        n = lambda s: random.gauss(0, s)
        return [seq, self.dev_mono_ns(t_laptop_ns), self.dev_wall_ns(t_laptop_ns),
                round(2 * math.sin(ph) + n(0.1), 4),
                round(9.81 + 3 * math.cos(ph) + impact + n(0.1), 4),
                round(1.5 * math.sin(2 * ph) + n(0.1), 4),
                round(150 * math.sin(ph) + n(2), 3),
                round(20 * math.sin(2 * ph) + n(2), 3),
                round(10 * math.cos(ph) + n(2), 3)]

    def publish_batch(self, rows):
        a = self.args
        if a.drop and random.random() < a.drop:
            return
        payload = json.dumps({
            "device": "unoq-shin", "session": self.session, "rate_hz": self.rate_hz,
            "fields": ["ax", "ay", "az", "gx", "gy", "gz"],
            "units": {"accel": "m/s^2", "gyro": "deg/s"}, "samples": rows})
        if a.reorder and self._held is None and random.random() < a.reorder:
            self._held = payload  # send after the next batch
            return
        self.client.publish(self.topics["raw"], payload)
        if self._held is not None:
            self.client.publish(self.topics["raw"], self._held)
            self._held = None
        self.sent += len(rows)

    def publish_status(self, buffered):
        elapsed = (time.monotonic_ns() - self.t0_ns) / 1e9
        self.client.publish(self.topics["status"], json.dumps({
            "device": "unoq-shin", "online": True, "session": self.session,
            "rate_hz_measured": round(self.rate_hz * (1 + self.drift), 3),
            "samples_sent": self.sent, "samples_buffered": buffered,
            "reconnects": self.reconnects, "accel_range_g": 8, "gyro_range_dps": 1000,
            "uptime_s": round(elapsed, 1)}), qos=1, retain=True)

    def run(self):
        a = self.args
        m = self.cfg["mqtt"]
        self.client.connect(a.broker or m["host"], m["port"], keepalive=m["keepalive_s"])
        self.client.loop_start()
        print(f"[fake] session {self.session}  {self.rate_hz:g} Hz, batches of {self.batch_size}")

        seq = 0
        rows, held = [], []
        next_t = time.monotonic_ns()
        next_status = next_t
        in_outage = False
        try:
            while a.duration is None or (time.monotonic_ns() - self.t0_ns) / 1e9 < a.duration:
                now = time.monotonic_ns()
                if now < next_t:
                    time.sleep(min((next_t - now) / 1e9, 0.005))
                    continue
                period_ns = int(1e9 / self.rate_hz)
                elapsed = (now - self.t0_ns) / 1e9
                outage = a.outage_at is not None and a.outage_at <= elapsed < a.outage_at + a.outage_for
                if outage and not in_outage:
                    print(f"[fake] outage start ({'buffering' if a.buffer else 'dropping'})")
                    self.client.disconnect()
                    self.client.loop_stop()
                elif in_outage and not outage:
                    self.client.reconnect()
                    self.client.loop_start()
                    self.reconnects += 1
                    print(f"[fake] outage over, sending {len(held)} buffered samples")
                    for i in range(0, len(held), self.batch_size):
                        self.publish_batch(held[i:i + self.batch_size])
                    held = []
                in_outage = outage

                if self.running:
                    rows.append(self.sample(seq, next_t))
                    seq += 1
                    if len(rows) >= self.batch_size:
                        if outage:
                            if a.buffer:
                                held.extend(rows)
                        else:
                            self.publish_batch(rows)
                        rows = []
                next_t += period_ns

                if now >= next_status and not outage:
                    self.publish_status(len(held))
                    next_status = now + 1_000_000_000
        except KeyboardInterrupt:
            pass
        finally:
            self.client.publish(self.topics["status"], json.dumps(
                {"device": "unoq-shin", "online": False}), qos=1, retain=True)
            time.sleep(0.2)
            self.client.loop_stop()
            self.client.disconnect()
            print(f"[fake] stopped after seq {seq}, sent {self.sent}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default=None)
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--outage-at", type=float, default=None)
    p.add_argument("--outage-for", type=float, default=5.0)
    p.add_argument("--buffer", action="store_true")
    p.add_argument("--reorder", type=float, default=0.0)
    p.add_argument("--drop", type=float, default=0.0)
    FakeDevice(load_config(), p.parse_args()).run()


if __name__ == "__main__":
    main()
