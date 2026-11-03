"""A stand-in for arduino-router + the shin_imu sketch, for tests without hardware.

Listens on a Unix socket and speaks the router's MessagePack-RPC dialect:
answers `$/register`, the sketch's methods (now_ns, mcu_stats, stream,
set_rate, set_synthetic, set_led) and sends `imu_batch` notifications in
the sketch's binary format to whoever registered it.

The simulated MCU clock runs `skew_ppm` fast with an arbitrary offset, so
the clock-mapping code has something real to correct.
"""

import math
import socket
import struct
import threading
import time

import msgpack

RECORD = struct.Struct("<IQ6h")


class FakeRouter:
    def __init__(self, path, rate_hz=100, skew_ppm=40.0, batch=10, rtt_ms=1.0):
        self.path = path
        self.rate_hz = rate_hz
        self.skew = skew_ppm * 1e-6
        self.batch = batch
        self.rtt_s = rtt_ms / 1000
        self.boot_ns = time.monotonic_ns() - 3_000_000_000  # "MCU booted 3 s ago"
        self.streaming = True
        self.led = None
        self.drop_ticks = set()  # tick numbers to skip (simulated lost samples)
        self.calls = []
        self._clients = []
        self._sample_client = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def mcu_ns(self, linux_ns=None):
        linux_ns = time.monotonic_ns() if linux_ns is None else linux_ns
        return int((linux_ns - self.boot_ns) * (1 + self.skew))

    def linux_ns(self, mcu_ns):
        return int(mcu_ns / (1 + self.skew)) + self.boot_ns

    # ---- server ---------------------------------------------------------
    def start(self):
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.path)
        self._srv.listen(4)
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._sampler, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        self._srv.close()
        for c in self._clients:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.close()

    def _accept(self):
        while not self._stop.is_set():
            try:
                c, _ = self._srv.accept()
            except OSError:
                return
            self._clients.append(c)
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _send(self, c, msg):
        with self._lock:
            try:
                c.sendall(msgpack.packb(msg))
            except OSError:
                pass

    def _serve(self, c):
        up = msgpack.Unpacker()
        while not self._stop.is_set():
            try:
                data = c.recv(65536)
            except OSError:
                return
            if not data:
                if self._sample_client is c:
                    self._sample_client = None
                return
            up.feed(data)
            for msg in up:
                if msg[0] == 0:
                    _, msgid, method, params = msg
                    self.calls.append((method, params))
                    if method == "$/register":
                        if params[0] == "imu_batch":
                            self._sample_client = c
                        self._send(c, [1, msgid, None, True])
                    else:
                        if self.rtt_s:
                            time.sleep(self.rtt_s / 2)
                        result = self._mcu(method, params)
                        if self.rtt_s:
                            time.sleep(self.rtt_s / 2)
                        self._send(c, [1, msgid, None, result])
                elif msg[0] == 2:
                    self.calls.append((msg[1], msg[2]))
                    self._mcu(msg[1], msg[2])

    def _mcu(self, method, params):
        if method == "now_ns":
            return self.mcu_ns()
        if method == "mcu_stats":
            return (f"fw=shin_imu/1;ok=1;src=imu;bus=Wire2;addr=0x68;who=0x68;accel_cfg=0x18;gyro_cfg=0x18;"
                    f"dlpf=0x03;rate={self.rate_hz};streaming={int(self.streaming)};late=0;i2c_err=0;"
                    f"overflow=0;sent=0;queued=0")
        if method == "stream":
            self.streaming = bool(params[0])
            return None
        if method == "set_rate":
            if params[0] not in (25, 50, 100, 200):
                return False
            self.rate_hz = params[0]
            return True
        if method == "set_led":
            self.led = params[0]
            return None
        return None

    # ---- sample generator (the "timer") ---------------------------------
    def _sampler(self):
        tick = 0
        buf = []
        next_mcu = self.mcu_ns()
        while not self._stop.is_set():
            period_mcu = 1_000_000_000 // self.rate_hz
            delay = (self.linux_ns(next_mcu) - time.monotonic_ns()) / 1e9
            if delay > 0:
                time.sleep(delay)
            t = next_mcu
            next_mcu += period_mcu
            if not self.streaming:
                if buf:
                    self._flush(buf)
                    buf = []
                continue
            this = tick
            tick += 1
            if this in self.drop_ticks:
                continue
            ph = 2 * math.pi * this / 100
            raw = (int(1000 * math.sin(ph)), 0, 2048, 0, 0, int(800 * math.cos(ph)))
            buf.append(RECORD.pack(this, t, *raw))
            if len(buf) == self.batch:
                self._flush(buf)
                buf = []

    def _flush(self, buf):
        c = self._sample_client
        if c is not None:
            self._send(c, [2, "imu_batch", [b"".join(buf)]])
