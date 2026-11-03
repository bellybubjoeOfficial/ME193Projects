"""Load the streamer config (TOML). See config/streamer.toml for every key."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

VALID_RATES = (25, 50, 100, 200)  # must match RATES[] in sketch/sketch.ino


@dataclass(frozen=True)
class Config:
    broker_host: str
    broker_port: int = 1883
    keepalive_s: int = 5
    client_id: str = "unoq-shin-streamer"
    username: str | None = None
    password: str | None = None
    device: str = "unoq-shin"
    topic_prefix: str = "wearable"
    rate_hz: int = 100
    batch_size: int = 10
    qos: int = 1
    max_inflight: int = 20
    buffer_seconds: float = 300.0
    autostart: bool = True
    synthetic_if_no_imu: bool = False
    router_socket: str = "/var/run/arduino-router.sock"

    def topic(self, name):
        return f"{self.topic_prefix}/{name}"


def load(path):
    raw = tomllib.loads(Path(path).read_text())
    mqtt, stream, dev, router = (raw.get(k, {}) for k in ("mqtt", "stream", "device", "router"))
    if not mqtt.get("host"):
        raise ValueError(f"{path}: [mqtt] host is required")
    cfg = Config(
        broker_host=mqtt["host"],
        broker_port=int(mqtt.get("port", 1883)),
        keepalive_s=int(mqtt.get("keepalive_s", 5)),
        client_id=mqtt.get("client_id", "unoq-shin-streamer"),
        username=mqtt.get("username") or None,
        # Secrets: prefer the environment (systemd EnvironmentFile) over the file.
        password=os.environ.get("SHIN_IMU_MQTT_PASSWORD") or mqtt.get("password") or None,
        device=dev.get("name", "unoq-shin"),
        topic_prefix=mqtt.get("topic_prefix", "wearable"),
        rate_hz=int(stream.get("rate_hz", 100)),
        batch_size=int(stream.get("batch_size", 10)),
        qos=int(mqtt.get("qos", 1)),
        max_inflight=int(mqtt.get("max_inflight", 20)),
        buffer_seconds=float(stream.get("buffer_seconds", 300)),
        autostart=bool(stream.get("autostart", True)),
        synthetic_if_no_imu=bool(stream.get("synthetic_if_no_imu", False)),
        router_socket=router.get("socket", "/var/run/arduino-router.sock"),
    )
    if cfg.rate_hz not in VALID_RATES:
        raise ValueError(f"{path}: rate_hz must be one of {VALID_RATES}")
    if cfg.buffer_seconds < 60:
        raise ValueError(f"{path}: buffer_seconds must be >= 60")
    return cfg
