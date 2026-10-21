"""
fake_unoq.py -- pretends to be the UNO Q, following MQTT_CONTRACT.md.

Publishes a fake distance that sweeps 100..800 mm at 20 Hz (with a
null every 50th reading to exercise "out of range"), publishes status with
a Last Will, and echoes pings. Use it to test the laptop side without the
real board, or to check the Q's broker from the laptop.

Usage (from ME193Sumo/):
    python tests/fake_unoq.py                               # broker from config.py
    python tests/fake_unoq.py --broker test.mosquitto.org   # public broker
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from mqttlib import MQTTClient
from unoq_link import PING_TOPIC, PONG_TOPIC, STATUS_TOPIC, sensor_topic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default=config.MQTT_BROKER)
    p.add_argument("--hz", type=float, default=20.0)
    args = p.parse_args()

    client = MQTTClient(broker=args.broker, port=config.MQTT_PORT)
    client._client.will_set(STATUS_TOPIC, json.dumps({"state": "offline"}), qos=1, retain=True)
    client.connect()
    client.publish(STATUS_TOPIC, json.dumps({"state": "online", "sensors": ["distance"],
                                             "version": "fake"}), qos=1, retain=True)
    client.subscribe(PING_TOPIC, lambda topic, payload: client.publish(PONG_TOPIC, payload))
    print(f"fake UNO Q publishing on {args.broker} at {args.hz:g} Hz (Ctrl+C to stop)")

    seq = 0
    start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - start
            mm = None if seq % 50 == 49 else int(450 + 350 * math.sin(t))
            client.publish(sensor_topic("distance"),
                           json.dumps({"seq": seq, "t": time.time(), "mm": mm}))
            seq += 1
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        client.publish(STATUS_TOPIC, json.dumps({"state": "offline"}), qos=1, retain=True)
        time.sleep(0.2)
        client.disconnect()


if __name__ == "__main__":
    main()
