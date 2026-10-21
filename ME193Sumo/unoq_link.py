"""
unoq_link.py

Laptop side of the UNO Q sensor link over MQTT (see MQTT_CONTRACT.md).

- `parse_message()` and `SensorCache` are pure (time is passed in) and
  unit tested in tests/test_unoq_link.py.
- `UnoQLink` is the thin MQTT layer: subscribes to the Q's topics, feeds
  the cache from paho's background thread, and measures round-trip
  latency with ping/pong.
"""

import json
import math
import threading
import time

import config

PREFIX = "me193sumo"
SENSORS = ("distance", "imu")
STATUS_TOPIC = f"{PREFIX}/unoq/status"
PING_TOPIC = f"{PREFIX}/laptop/ping"
PONG_TOPIC = f"{PREFIX}/unoq/pong"


def sensor_topic(name):
    return f"{PREFIX}/unoq/{name}"


# ── Parsing (pure) ───────────────────────────────────────────────────────────

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return (_is_int(v) or isinstance(v, float)) and math.isfinite(v)


def _validate_distance(msg):
    mm = msg.get("mm", "missing")
    if mm is not None and not (_is_int(mm) and mm >= 0):
        raise ValueError(f"distance.mm must be a non-negative int or null, got {mm!r}")


def _validate_imu(msg):
    for key in ("ax", "ay", "az", "gx", "gy", "gz"):
        if not _is_num(msg.get(key)):
            raise ValueError(f"imu.{key} must be a number, got {msg.get(key)!r}")


_VALIDATORS = {"distance": _validate_distance, "imu": _validate_imu}


def parse_message(sensor, payload):
    """Payload string → validated dict. Raises ValueError if malformed."""
    try:
        msg = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"bad JSON: {e}") from None
    if not isinstance(msg, dict):
        raise ValueError("payload must be a JSON object")
    if not _is_int(msg.get("seq")):
        raise ValueError(f"seq must be an int, got {msg.get('seq')!r}")
    if not _is_num(msg.get("t")):
        raise ValueError(f"t must be a number, got {msg.get('t')!r}")
    _VALIDATORS[sensor](msg)
    return msg


def parse_status(payload):
    """Status payload → state string ("online" / "offline"), or None if malformed."""
    try:
        msg = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    state = msg.get("state") if isinstance(msg, dict) else None
    return state if isinstance(state, str) else None


# ── Cache (pure) ─────────────────────────────────────────────────────────────

class _Track:
    def __init__(self):
        self.reading = None
        self.received_at = -math.inf
        self.count = 0
        self.dropped = 0
        self.missed = 0      # messages skipped according to seq gaps
        self.first_at = None


class SensorCache:
    """Latest reading per sensor + link-health counters."""

    def __init__(self, stale_s=None):
        self.stale_s = config.SENSOR_STALE_S if stale_s is None else stale_s
        self._tracks = {name: _Track() for name in SENSORS}

    def on_message(self, sensor, payload, now):
        """Feed one raw MQTT payload. Returns the parsed dict, or None if dropped."""
        track = self._tracks[sensor]
        try:
            msg = parse_message(sensor, payload)
        except ValueError:
            track.dropped += 1
            return None
        if track.reading is not None:
            gap = msg["seq"] - track.reading["seq"]
            if gap > 1:
                track.missed += gap - 1
            # gap <= 0: publisher restarted (seq reset) -- just carry on
        track.reading = msg
        track.received_at = now
        track.count += 1
        if track.first_at is None:
            track.first_at = now
        return msg

    def latest(self, sensor, now):
        """Most recent valid reading, or None if never received / stale."""
        track = self._tracks[sensor]
        if now - track.received_at > self.stale_s:
            return None
        return track.reading

    def distance_mm(self, now):
        """Distance in mm, or None if unknown (stale, never received, or out of range)."""
        reading = self.latest("distance", now)
        return None if reading is None else reading["mm"]

    def age(self, sensor, now):
        return now - self._tracks[sensor].received_at

    def stats(self, sensor, now):
        t = self._tracks[sensor]
        span = (now - t.first_at) if t.first_at is not None else 0.0
        rate = (t.count - 1) / span if t.count > 1 and span > 0 else 0.0
        return {"count": t.count, "dropped": t.dropped, "missed": t.missed,
                "rate_hz": rate, "age_s": now - t.received_at}


# ── MQTT layer ───────────────────────────────────────────────────────────────

class UnoQLink:
    def __init__(self, broker=None, port=None):
        from mqttlib import MQTTClient   # imported here so the pure parts don't need paho
        self.cache = SensorCache()
        self.status = None
        self._lock = threading.Lock()
        self._pongs = {}
        self._ping_id = 0
        self._client = MQTTClient(broker=broker or config.MQTT_BROKER,
                                  port=port or config.MQTT_PORT)

    def connect(self, timeout=5):
        self._client.connect(timeout=timeout)
        if not self._client._connected.is_set():
            raise ConnectionError(f"Couldn't reach MQTT broker at "
                                  f"{self._client.broker}:{self._client.port}")
        for name in SENSORS:
            self._client.subscribe(sensor_topic(name), self._sensor_cb(name))
        self._client.subscribe(STATUS_TOPIC, self._on_status, qos=1)
        self._client.subscribe(PONG_TOPIC, self._on_pong)

    def _sensor_cb(self, name):
        def cb(topic, payload):
            with self._lock:
                self.cache.on_message(name, payload, time.monotonic())
        return cb

    def _on_status(self, topic, payload):
        self.status = parse_status(payload)

    def _on_pong(self, topic, payload):
        try:
            ping_id = json.loads(payload)["id"]
        except (ValueError, KeyError, TypeError):
            return
        with self._lock:
            if ping_id in self._pongs:
                self._pongs[ping_id] = time.monotonic()

    def ping(self, timeout=1.0):
        """Round-trip time in seconds via the Q's pong echo, or None on timeout."""
        with self._lock:
            self._ping_id += 1
            ping_id = self._ping_id
            self._pongs[ping_id] = None
        sent = time.monotonic()
        self._client.publish(PING_TOPIC, json.dumps({"id": ping_id, "t": time.time()}))
        while time.monotonic() - sent < timeout:
            with self._lock:
                got = self._pongs[ping_id]
            if got is not None:
                with self._lock:
                    del self._pongs[ping_id]
                return got - sent
            time.sleep(0.005)
        with self._lock:
            del self._pongs[ping_id]
        return None

    def distance_mm(self):
        with self._lock:
            return self.cache.distance_mm(time.monotonic())

    def stats(self, sensor):
        with self._lock:
            return self.cache.stats(sensor, time.monotonic())

    def close(self):
        self._client.disconnect()
