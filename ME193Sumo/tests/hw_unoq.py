"""
H4.1 -- can the laptop receive the UNO Q's sensor data?

Connects to the broker, listens for --seconds, pings the Q, then prints a
PASS/FAIL report and exits 0 (all pass) or 1.

Usage (from ME193Sumo/):
    python tests/hw_unoq.py                    # real Q (broker from config.py)
    python tests/hw_unoq.py --live             # also stream readings while listening
    python tests/hw_unoq.py --broker test.mosquitto.org   # with tests/fake_unoq.py

Pass criteria (MQTT_CONTRACT.md):
  - broker reachable
  - status topic says "online"
  - distance arrives at >= 15 Hz with no malformed messages
  - < 5% of messages missed (seq gaps)
  - median ping round trip < 100 ms
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from unoq_link import UnoQLink

MIN_RATE_HZ = 15.0
MAX_MISSED_FRACTION = 0.05
MAX_RTT_S = 0.100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default=config.MQTT_BROKER)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--live", action="store_true")
    args = p.parse_args()

    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"Connecting to MQTT broker {args.broker}:{config.MQTT_PORT} ...")
    link = UnoQLink(broker=args.broker)
    try:
        link.connect()
    except Exception as e:
        print(f"  [FAIL] broker reachable: {e}")
        print("\nIs Mosquitto running on the Q and listening on 0.0.0.0:1883?")
        return 1

    print(f"Listening for {args.seconds:g} s ...")
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        if args.live:
            mm = link.distance_mm()
            print(f"\r  distance = {('---' if mm is None else mm):>5} mm   ", end="", flush=True)
        time.sleep(0.05)
    if args.live:
        print()

    rtts = [r for r in (link.ping() for _ in range(10)) if r is not None]
    s = link.stats("distance")
    total = s["count"] + s["missed"]

    print("\nResults")
    check("broker reachable", True, f"{args.broker}:{config.MQTT_PORT}")
    check("status online", link.status == "online", f"status = {link.status!r}")
    check("distance received", s["count"] > 0, f"{s['count']} messages")
    check("distance rate", s["rate_hz"] >= MIN_RATE_HZ, f"{s['rate_hz']:.1f} Hz (need >= {MIN_RATE_HZ:g})")
    check("no malformed messages", s["dropped"] == 0, f"{s['dropped']} dropped")
    check("few missed messages", total > 0 and s["missed"] / total <= MAX_MISSED_FRACTION,
          f"{s['missed']} missed of {total}")
    if rtts:
        med = statistics.median(rtts)
        check("ping round trip", med < MAX_RTT_S,
              f"median {med * 1000:.0f} ms, max {max(rtts) * 1000:.0f} ms ({len(rtts)}/10 answered)")
    else:
        check("ping round trip", False, "no pongs -- is the Q echoing me193sumo/laptop/ping?")
    print(f"  last distance: {link.distance_mm()} mm")

    link.close()
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
