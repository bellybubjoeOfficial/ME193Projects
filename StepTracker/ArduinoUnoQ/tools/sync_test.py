#!/usr/bin/env python3
"""NTP-style ping test: send N pings to wearable/sync/ping, time the pongs.

Run it on the laptop (that is the clock being synced against):

    python tools/sync_test.py --host 127.0.0.1 --n 20

For each exchange (all wall-clock ns):
    t1 = t_laptop_ns (ping sent)      t2 = t_dev_recv_ns
    t3 = t_dev_send_ns                t4 = pong received on the laptop
    rtt    = (t4 - t1) - (t3 - t2)
    offset = ((t2 - t1) + (t3 - t4)) / 2      device clock minus laptop clock
"""

import argparse
import json
import statistics
import sys
import threading
import time
import uuid

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--prefix", default="wearable")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=3.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pongs = {}
    got = threading.Event()
    subscribed = threading.Event()

    def on_connect(c, ud, flags, rc, props):
        c.subscribe(f"{args.prefix}/sync/pong", qos=0)

    def on_subscribe(c, ud, mid, rcs, props):
        subscribed.set()

    def on_message(c, ud, msg):
        t4 = time.time_ns()
        p = json.loads(msg.payload)
        pongs[p["id"]] = (p, t4)
        got.set()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"shin-sync-{uuid.uuid4().hex[:6]}")
    c.on_connect, c.on_subscribe, c.on_message = on_connect, on_subscribe, on_message
    c.connect(args.host, args.port, keepalive=10)
    c.loop_start()
    if not subscribed.wait(5):
        sys.exit("could not subscribe")

    rows = []
    for i in range(args.n):
        pid = f"{uuid.uuid4().hex[:8]}-{i}"
        got.clear()
        t1 = time.time_ns()
        c.publish(f"{args.prefix}/sync/ping", json.dumps({"id": pid, "t_laptop_ns": t1}), qos=0)
        deadline = time.monotonic() + args.timeout
        while pid not in pongs and time.monotonic() < deadline:
            got.wait(0.05)
        if pid not in pongs:
            rows.append({"id": pid, "lost": True})
        else:
            p, t4 = pongs[pid]
            assert p["t_laptop_ns"] == t1
            t2, t3 = p["t_dev_recv_ns"], p["t_dev_send_ns"]
            rows.append({"id": pid, "rtt_ms": ((t4 - t1) - (t3 - t2)) / 1e6,
                         "offset_ms": ((t2 - t1) + (t3 - t4)) / 2e6,
                         "device_hold_us": (t3 - t2) / 1e3})
        time.sleep(args.interval)
    c.loop_stop()
    c.disconnect()

    ok = [r for r in rows if not r.get("lost")]
    summary = {"pings": len(rows), "pongs": len(ok)}
    if ok:
        rtts = sorted(r["rtt_ms"] for r in ok)
        offs = [r["offset_ms"] for r in ok]
        best = min(ok, key=lambda r: r["rtt_ms"])
        summary.update({
            "rtt_ms_min": round(rtts[0], 3), "rtt_ms_median": round(statistics.median(rtts), 3),
            "rtt_ms_max": round(rtts[-1], 3), "rtt_ms_mean": round(statistics.fmean(rtts), 3),
            "offset_ms_median": round(statistics.median(offs), 3),
            "offset_ms_at_min_rtt": round(best["offset_ms"], 3),
            "offset_ms_spread": round(max(offs) - min(offs), 3),
            "device_hold_us_median": round(statistics.median(r["device_hold_us"] for r in ok), 1),
        })
    if args.json:
        print(json.dumps({"summary": summary, "pings": rows}, indent=2))
    else:
        for r in rows:
            if r.get("lost"):
                print(f"{r['id']}: LOST")
            else:
                print(f"{r['id']}: rtt {r['rtt_ms']:7.3f} ms  offset {r['offset_ms']:+8.3f} ms  "
                      f"device hold {r['device_hold_us']:.0f} us")
        for k, v in summary.items():
            print(f"{k:24s} {v}")
    sys.exit(0 if len(ok) == len(rows) else 1)


if __name__ == "__main__":
    main()
