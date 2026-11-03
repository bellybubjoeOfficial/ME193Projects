"""Minimal MessagePack-RPC client for the arduino-router Unix socket.

This is the same protocol the App Lab `Bridge` class speaks (see
arduino/app_utils/bridge.py in the python-apps-base image), reimplemented
here so the streamer can run as a plain systemd service on the Linux host
instead of inside an App Lab container:

    request      [0, msgid, method, params]
    response     [1, msgid, error, result]
    notification [2, method, params]

Methods we provide to the MCU are registered with the router via
`$/register`, and re-registered after every reconnect.
"""

import logging
import queue
import socket
import threading

import msgpack

log = logging.getLogger("router")

ROUTE_ALREADY_EXISTS_ERR = 0x05
RECONNECT_DELAY_S = 2.0


class RpcError(RuntimeError):
    """The far side answered with an error (e.g. method not found on the MCU)."""


class RouterClient:
    def __init__(self, path, on_connect=None):
        self.path = path
        self.on_connect = on_connect  # called (in its own thread) after each (re)connect
        self._handlers = {}
        self._pending = {}  # msgid -> Queue(1)
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._next_id = 0
        self._sock = None
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # ---- lifecycle -----------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="router", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._close()
        if self._thread:
            self._thread.join(timeout=RECONNECT_DELAY_S + 1)

    @property
    def connected(self):
        return self._connected.is_set()

    def wait_connected(self, timeout):
        return self._connected.wait(timeout)

    # ---- API -----------------------------------------------------------
    def provide(self, method, handler):
        """Handle `method` calls/notifications coming from the MCU."""
        self._handlers[method] = handler
        if self.connected:
            self.call("$/register", method, timeout=2)

    def notify(self, method, *params):
        self._send([2, method, list(params)])

    def call(self, method, *params, timeout=2.0):
        with self._lock:
            self._next_id = (self._next_id + 1) % 2**32
            msgid = self._next_id
            q = self._pending[msgid] = queue.Queue(maxsize=1)
        try:
            self._send([0, msgid, method, list(params)])
            ok, value = q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{method}: no reply in {timeout}s") from None
        finally:
            with self._lock:
                self._pending.pop(msgid, None)
        if not ok:
            raise RpcError(f"{method}: {value}")
        return value

    # ---- internals -----------------------------------------------------
    def _send(self, msg):
        data = msgpack.packb(msg)
        with self._send_lock:
            if self._sock is None:
                raise ConnectionError("not connected to arduino-router")
            try:
                self._sock.sendall(data)
            except OSError as e:
                raise ConnectionError(f"router send failed: {e}") from e

    def _close(self):
        self._connected.clear()
        with self._send_lock:
            if self._sock is not None:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._sock.close()
                self._sock = None
        with self._lock:
            for q in self._pending.values():
                q.put_nowait((False, "connection to router lost"))
            self._pending.clear()

    def _run(self):
        while not self._stop.is_set():
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.path)
            except OSError as e:
                log.warning("cannot connect to %s: %s", self.path, e)
                self._stop.wait(RECONNECT_DELAY_S)
                continue
            with self._send_lock:
                self._sock = s
            self._connected.set()
            log.info("connected to arduino-router at %s", self.path)
            threading.Thread(target=self._after_connect, name="router-register", daemon=True).start()
            self._read_loop(s)
            self._close()
            if not self._stop.is_set():
                log.warning("router connection lost; reconnecting")
                self._stop.wait(RECONNECT_DELAY_S)

    def _after_connect(self):
        for method in list(self._handlers):
            try:
                self.call("$/register", method, timeout=2)
            except Exception as e:  # noqa: BLE001 - keep going, log it
                log.error("could not register %s with the router: %s", method, e)
        if self.on_connect:
            self.on_connect()

    def _read_loop(self, s):
        unpacker = msgpack.Unpacker()
        while not self._stop.is_set():
            try:
                data = s.recv(65536)
            except OSError:
                return
            if not data:
                return
            unpacker.feed(data)
            for msg in unpacker:
                try:
                    self._dispatch(msg)
                except Exception:  # noqa: BLE001 - one bad message must not kill the reader
                    log.exception("error handling router message")

    def _dispatch(self, msg):
        kind = msg[0]
        if kind == 1:  # response
            _, msgid, err, result = msg
            with self._lock:
                q = self._pending.get(msgid)
            if q is None:
                return
            if err is not None and not (isinstance(err, list) and err and err[0] == ROUTE_ALREADY_EXISTS_ERR):
                q.put_nowait((False, err))
            else:
                q.put_nowait((True, result))
        elif kind == 2:  # notification from the MCU
            _, method, params = msg
            handler = self._handlers.get(method)
            if handler:
                handler(*params)
        elif kind == 0:  # request from the MCU: we don't serve any calls
            _, msgid, method, _params = msg
            self._send([1, msgid, [0xFE, f"method not found: {method}"], None])
