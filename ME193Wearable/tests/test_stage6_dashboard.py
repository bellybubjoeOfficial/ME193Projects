"""Stage 6 dashboard tests: the HUD web server (scripts/dashboard.py), against a temp control folder."""

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from wearable import config, recording

spec = importlib.util.spec_from_file_location("dashboard", config.ROOT / "scripts" / "dashboard.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONTROL_DIR", tmp_path)
    dashboard.HISTORY.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()


def call(url, method="GET"):
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_serves_the_page(server):
    url, _ = server
    code, body = call(url + "/")
    assert code == 200 and b"<title>Activity HUD</title>" in body
    assert call(url + "/nope")[0] == 404


def test_status_includes_history_and_classes(server):
    url, ctl = server
    recording.write_status(ctl, "live", model="model2", label="walk", raw_label="run",
                           proba={"walk": 0.6, "run": 0.4}, vote_fraction=0.67, n=1)
    dashboard.collect_once(None)
    d = json.loads(call(url + "/api/status")[1])
    assert d["status"]["label"] == "walk" and d["classes"] == config.CLASSES
    assert [h["label"] for h in d["history"]] == ["walk"]


def test_history_only_grows_on_new_live_predictions(server):
    _, ctl = server
    recording.write_status(ctl, "live", model="model1", label="run", proba={}, n=1)
    last = dashboard.collect_once(None)
    assert dashboard.collect_once(last) == last          # same status file: not added twice
    recording.write_status(ctl, "recording", activity="walk")
    dashboard.collect_once(last)
    assert len(dashboard.HISTORY) == 1                    # non-live states aren't predictions


def test_engage_writes_live_request(server):
    url, ctl = server
    assert call(url + "/api/live/model1", "POST")[0] == 200
    assert recording.take_live_request(ctl) == "model1"
    assert call(url + "/api/live/model7", "POST")[0] == 400


def test_never_interrupts_a_recording(server):
    url, ctl = server
    recording.write_status(ctl, "recording", activity="run")
    assert call(url + "/api/live/model2", "POST")[0] == 409
    assert call(url + "/api/stop", "POST")[0] == 409      # stop.json would end the recording early
    assert not recording.take_stop(ctl) and recording.take_live_request(ctl) is None


def test_stop_only_in_live_mode(server):
    url, ctl = server
    recording.write_status(ctl, "live", model="model2")
    assert call(url + "/api/stop", "POST")[0] == 200
    assert recording.take_stop(ctl)
