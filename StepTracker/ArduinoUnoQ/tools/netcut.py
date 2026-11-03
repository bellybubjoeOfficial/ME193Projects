#!/usr/bin/env python3
"""TCP proxy that can black-hole a link: simulates the device losing Wi-Fi.

Point the streamer's [mqtt] host/port at this proxy; it forwards to the real
broker. After --cut-after seconds it silently stops forwarding for
--cut-for seconds (no FIN/RST, packets just vanish, like a dead radio
link), so the streamer only notices through its MQTT keepalive. Then it
closes the dead connections and forwards again.

    python tools/netcut.py --listen 18831 --to 127.0.0.1:1883 --cut-after 60 --cut-for 30
"""

import argparse
import socket
import sys
import threading
import time


class Proxy:
    def __init__(self, listen_port, target):
        self.target = target
        self.cut = threading.Event()
        self.conns = []
        self.lock = threading.Lock()
        self.srv = socket.create_server(("127.0.0.1", listen_port), reuse_port=False)

    def serve(self):
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            c, _ = self.srv.accept()
            if self.cut.is_set():
                with self.lock:
                    self.conns.append((c, None))  # hold it open, never answer (dead link)
                continue
            try:
                u = socket.create_connection(self.target)
            except OSError:
                c.close()
                continue
            with self.lock:
                self.conns.append((c, u))
            for a, b in ((c, u), (u, c)):
                threading.Thread(target=self._pipe, args=(a, b), daemon=True).start()

    def _pipe(self, src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                while self.cut.is_set():
                    time.sleep(0.05)  # data is dropped: the link is "down"
                    data = b""
                if data:
                    dst.sendall(data)
        except OSError:
            pass
        for s in (src, dst):
            try:
                s.close()
            except OSError:
                pass

    def start_cut(self):
        self.cut.set()

    def end_cut(self):
        # Connections that lived through the outage are dead: drop them.
        with self.lock:
            dead, self.conns = self.conns, []
        for c, u in dead:
            for s in (c, u):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
        self.cut.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--to", required=True, help="broker host:port")
    ap.add_argument("--cut-after", type=float, required=True)
    ap.add_argument("--cut-for", type=float, default=30)
    ap.add_argument("--hold", type=float, default=0, help="keep proxying this long after the cut")
    args = ap.parse_args()
    host, port = args.to.rsplit(":", 1)
    p = Proxy(args.listen, (host, int(port)))
    p.serve()
    print(f"[netcut] proxy :{args.listen} -> {args.to}; cutting in {args.cut_after} s", file=sys.stderr)
    time.sleep(args.cut_after)
    print(f"[netcut] {time.strftime('%H:%M:%S')} LINK DOWN for {args.cut_for} s", file=sys.stderr)
    p.start_cut()
    time.sleep(args.cut_for)
    p.end_cut()
    print(f"[netcut] {time.strftime('%H:%M:%S')} LINK UP", file=sys.stderr)
    time.sleep(args.hold if args.hold else 1e9)


if __name__ == "__main__":
    main()
