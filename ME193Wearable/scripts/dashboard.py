#!/usr/bin/env python3
"""Stages 6-7: live HUD web page (probability graph + big "WALK"/"RUN" banner).

Run ON THE Q (over SSH), next to the running App:

    python scripts/dashboard.py                 # then open http://UNOQ3.local:8000 on the laptop/phone
    python scripts/dashboard.py --port 8080

The page can start/stop live mode (Model 1 or 2) itself, or you can use
scripts/live.py as before; both talk to the App the same way.

How it works: the App writes each live prediction to data/control/status.json
(that's all live mode ever does). This small server, standard library only,
runs outside the App: it watches that file, keeps the last few minutes of
predictions in memory, and serves them as JSON to web/dashboard.html. The App
and its 100 Hz sampling aren't touched; closing this server changes nothing
on the board.
"""

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wearable import config, recording  # noqa: E402

PAGE = config.ROOT / "web" / "dashboard.html"
POLL_S = 0.25
HISTORY = deque(maxlen=300)          # ~6.4 min of predictions at one per 1.28 s
LOCK = threading.Lock()


def collect_once(last):
    """Append the current live prediction to HISTORY if it's new. Returns its timestamp."""
    st = recording.read_status(config.CONTROL_DIR) or {}
    stamp = st.get("updated_at")
    if st.get("state") == "live" and st.get("label") and stamp != last:
        with LOCK:
            HISTORY.append({"t": stamp, "label": st["label"], "raw_label": st.get("raw_label"),
                            "proba": st.get("proba", {}), "model": st.get("model")})
    return stamp


def watch_status():
    """Background thread: collect every NEW live prediction."""
    last = None
    while True:
        last = collect_once(last)
        time.sleep(POLL_S)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path.startswith("/api/status"):
            with LOCK:
                hist = list(HISTORY)
            return self._send(200, {"status": recording.read_status(config.CONTROL_DIR) or {},
                                    "history": hist, "classes": config.CLASSES, "now": time.time()})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        st = recording.read_status(config.CONTROL_DIR) or {}
        if self.path.startswith("/api/live/"):
            if st.get("state") in ("countdown", "recording"):
                return self._send(409, {"error": "a recording is running; live mode waits for it"})
            name = self.path.rsplit("/", 1)[-1]
            try:
                recording.write_live_request(config.CONTROL_DIR, name)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            with LOCK:
                HISTORY.clear()                 # new model, fresh graph
            return self._send(200, {"ok": True})
        if self.path == "/api/stop":
            # stop.json also ends RECORDINGS, so only send it while live mode is on.
            if st.get("state") != "live":
                return self._send(409, {"error": f"not in live mode (state: {st.get('state')})"})
            recording.write_stop(config.CONTROL_DIR)
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):     # keep the terminal quiet (it's polled 3x a second)
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    threading.Thread(target=watch_status, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    host = socket.gethostname()
    print(f"Activity HUD: http://{host}.local:{args.port}   (or http://<board IP>:{args.port})\n"
          f"Ctrl-C to quit (live mode on the board keeps running).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
