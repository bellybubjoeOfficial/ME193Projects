#!/usr/bin/env python3
"""Analyze a recording from tools/record.py: rate, seq gaps, jitter, peaks.

    python tools/analyze.py bench.jsonl [--rate 100] [--json]

Exit code 0 if every session passes: measured rate within +-1 % of nominal,
no seq gaps, no duplicates left after de-duplication, every non-final
message has exactly 10 samples.
"""

import argparse
import json
import math
import statistics
import sys

G = 9.80665


def load(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def gaps_of(seqs):
    """Sorted unique seqs -> list of (first_missing, n_missing)."""
    out = []
    for a, b in zip(seqs, seqs[1:]):
        if b - a > 1:
            out.append((a + 1, b - a - 1))
    return out


def analyze(path, nominal=None):
    sessions = {}
    statuses = []
    diags = []
    for rec in load(path):
        p = rec["payload"]
        if rec["topic"].endswith("imu/raw"):
            s = sessions.setdefault(p["session"], {"rows": [], "msg_sizes": [], "rates": set(), "arrivals": []})
            s["rows"].extend(p["samples"])
            s["msg_sizes"].append(len(p["samples"]))
            s["rates"].add(p["rate_hz"])
            s["arrivals"].append(rec["t_rx_ns"])
        elif rec["topic"].endswith("/status"):
            statuses.append(p)
        elif rec["topic"].endswith("/diag"):
            diags.append(p)

    report = {"file": path, "sessions": []}
    ok_all = bool(sessions)
    for sid, s in sessions.items():
        by_seq = {}
        dups = 0
        out_of_order = 0
        last = -1
        for r in s["rows"]:
            if r[0] in by_seq:
                dups += 1
                continue
            if r[0] < last:
                out_of_order += 1
            last = max(last, r[0])
            by_seq[r[0]] = r
        rows = [by_seq[k] for k in sorted(by_seq)]
        seqs = [r[0] for r in rows]
        gaps = gaps_of(seqs)
        t = [r[1] for r in rows]
        span_s = (t[-1] - t[0]) / 1e9 if len(t) > 1 else 0
        rate = (len(rows) - 1) / span_s if span_s else 0
        # Interval stats on consecutive seqs only, so a gap doesn't count as jitter.
        d = [(b[1] - a[1]) / 1e3 for a, b in zip(rows, rows[1:]) if b[0] == a[0] + 1]
        wall_minus_mono = [r[2] - r[1] for r in rows]
        acc = [math.sqrt(r[3] ** 2 + r[4] ** 2 + r[5] ** 2) for r in rows]
        nom = nominal or (min(s["rates"]) if len(s["rates"]) == 1 else None)
        peak_axis = {f: max(abs(r[i]) for r in rows) for i, f in enumerate(["ax", "ay", "az", "gx", "gy", "gz"], 3)}
        sat_a = sum(1 for r in rows if max(abs(v) for v in r[3:6]) >= 16 * G * 0.999)
        sat_g = sum(1 for r in rows if max(abs(v) for v in r[6:9]) >= 2000 * 0.999)
        sizes = s["msg_sizes"]
        sess = {
            "session": sid,
            "rate_hz_nominal": sorted(s["rates"]),
            "samples_unique": len(rows),
            "duplicates_received": dups,
            "out_of_order_received": out_of_order,
            "messages": len(sizes),
            "msgs_not_10": sum(1 for n in sizes[:-1] if n != 10),
            "seq_first_last": [seqs[0], seqs[-1]],
            "seq_gaps": len(gaps),
            "seq_missing": sum(n for _, n in gaps),
            "gap_examples": gaps[:5],
            "duration_s": round(span_s, 2),
            "rate_hz_measured": round(rate, 4),
            "rate_err_pct": round(100 * (rate - nom) / nom, 4) if nom and rate else None,
            "interval_mean_us": round(statistics.fmean(d), 2) if d else None,
            "interval_std_us": round(statistics.pstdev(d), 2) if len(d) > 1 else None,
            "interval_min_max_us": [round(min(d), 1), round(max(d), 1)] if d else None,
            "wall_minus_mono_spread_ms": round((max(wall_minus_mono) - min(wall_minus_mono)) / 1e6, 3),
            "peak_abs": {k: round(v, 3) for k, v in peak_axis.items()},
            "peak_accel_norm_ms2": round(max(acc), 3),
            "peak_accel_norm_g": round(max(acc) / G, 3),
            "samples_at_accel_full_scale": sat_a,
            "samples_at_gyro_full_scale": sat_g,
            "latency_ms_median": round(statistics.median(
                (a - r[2]) / 1e6 for a, r in zip(s["arrivals"], (s["rows"][i] for i in
                                                                 _last_row_idx(sizes)))), 1),
        }
        sess["pass"] = (sess["seq_gaps"] == 0 and sess["msgs_not_10"] == 0
                        and sess["rate_err_pct"] is not None and abs(sess["rate_err_pct"]) <= 1.0)
        ok_all &= sess["pass"]
        report["sessions"].append(sess)
    if statuses:
        last = statuses[-1]
        report["status_last"] = last
        report["status_count"] = len(statuses)
        report["status_buffered_max"] = max((x.get("samples_buffered") or 0) for x in statuses)
        report["status_reconnects_max"] = max((x.get("reconnects") or 0) for x in statuses)
        report["status_offline_seen"] = any(x.get("online") is False for x in statuses)
    if diags:
        report["diag_last"] = diags[-1]
    report["pass"] = ok_all
    return report


def _last_row_idx(sizes):
    i = 0
    for n in sizes:
        i += n
        yield i - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--rate", type=float, help="nominal rate (default: rate_hz in the messages)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args()
    rep = analyze(args.file, args.rate)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        for s in rep["sessions"]:
            print(f"session {s['session']}  {'PASS' if s['pass'] else 'FAIL'}")
            for k, v in s.items():
                if k not in ("session", "pass"):
                    print(f"  {k:30s} {v}")
        for k in ("status_count", "status_buffered_max", "status_reconnects_max", "status_offline_seen"):
            if k in rep:
                print(f"{k:32s} {rep[k]}")
        if "status_last" in rep:
            print(f"{'status_last':32s} {json.dumps(rep['status_last'])}")
        print("OVERALL", "PASS" if rep["pass"] else "FAIL")
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
