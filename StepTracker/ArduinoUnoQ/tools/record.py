#!/usr/bin/env python3
"""Record wearable/* MQTT traffic to a JSON-lines file (runs on laptop or board).

Uses a persistent session (clean_session=False, QoS 1) so that if the
broker is restarted mid-test, messages published while this recorder was
reconnecting are still delivered to it (needs Mosquitto `persistence true`,
the Debian/Ubuntu default; Homebrew/Windows installs may need it set).

    python tools/record.py --host 127.0.0.1 --seconds 600 --out bench.jsonl
"""

import argparse
import json
import sys
import threading
import time

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--prefix", default="wearable")
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--client-id", default="shin-recorder")
    args = ap.parse_args()

    out = open(args.out, "w")
    lock = threading.Lock()
    counts = {"raw": 0, "samples": 0, "other": 0}

    def on_connect(c, ud, flags, rc, props):
        print(f"[record] connected ({rc}), session_present={flags.session_present}", file=sys.stderr)
        c.subscribe([(f"{args.prefix}/imu/raw", 1), (f"{args.prefix}/status", 1),
                     (f"{args.prefix}/diag", 1), (f"{args.prefix}/cmd/ack", 1)])

    def on_disconnect(c, ud, flags, rc, props):
        print(f"[record] disconnected ({rc})", file=sys.stderr)

    def on_message(c, ud, msg):
        t_rx = time.time_ns()
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            payload = {"_unparsed": msg.payload.decode(errors="replace")}
        with lock:
            out.write(json.dumps({"t_rx_ns": t_rx, "topic": msg.topic, "retain": bool(msg.retain),
                                  "payload": payload}) + "\n")
            if msg.topic.endswith("imu/raw"):
                counts["raw"] += 1
                counts["samples"] += len(payload.get("samples", []))
            else:
                counts["other"] += 1

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.client_id, clean_session=False)
    c.on_connect, c.on_disconnect, c.on_message = on_connect, on_disconnect, on_message
    c.reconnect_delay_set(1, 2)
    c.connect_async(args.host, args.port, keepalive=10)
    c.loop_start()
    t_end = time.monotonic() + args.seconds
    try:
        while time.monotonic() < t_end:
            time.sleep(min(10, max(0.0, t_end - time.monotonic())))
            with lock:
                out.flush()
                print(f"[record] {round(args.seconds - (t_end - time.monotonic()))} s: "
                      f"{counts['raw']} raw msgs, {counts['samples']} samples", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    c.loop_stop()
    c.disconnect()
    out.close()


if __name__ == "__main__":
    main()
