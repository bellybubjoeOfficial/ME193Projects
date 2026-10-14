"""
score_mqtt.py

Publishes the live score in real time to the public test.mosquitto.org
broker, over MQTT -- same broker/library (mqttlib.py, reused from
ME193HW2) this course's other assignments already use. Topic is
ME193/Rogers/<name>, matching ME193_TwoComputer/config.py's
"ME193/<TEAM_NAME>/..." convention, just with this project's own name
instead of a team placeholder.

game.py calls connect_score_mqtt()/disconnect_score_mqtt() alongside its
other optional-hardware connections, and publish_score() every time the
score or miss count changes; nothing else needs to know whether the
broker is actually reachable -- if connect_score_mqtt() can't reach it,
it prints a warning and publish_score() silently no-ops, so the game
stays playable with no network at all (same "optional, never blocks or
crashes the game" philosophy as haptics.py).
"""

from mqttlib import MQTTClient

BROKER = "test.mosquitto.org"
PORT = 1883

# ME193/Rogers is the shared course namespace (see
# ME193_TwoComputer/config.py's GAME_TOPIC); this project's own name is
# appended after it, same as that file's TEAM_NAME convention, so this
# doesn't collide with anyone else publishing under ME193/Rogers on the
# same public broker.
SCORE_TOPIC = "ME193/Rogers/Shanmugam"

CONNECT_TIMEOUT_S = 5

_client = None
_connected = False


def connect_score_mqtt():
    """Connects to the broker. On failure, prints a warning and leaves
    publish_score() as a no-op -- the game still runs fine without it."""
    global _client, _connected
    print(f"[score-mqtt] Connecting to {BROKER}:{PORT} (topic {SCORE_TOPIC})...")
    try:
        _client = MQTTClient(broker=BROKER, port=PORT)
        _client.connect(timeout=CONNECT_TIMEOUT_S)
        _connected = True
        print("[score-mqtt] Connected.")
    except Exception as exc:
        print(f"[score-mqtt] Could not connect ({exc}) -- continuing without live score publishing.")
        _client = None
        _connected = False


def disconnect_score_mqtt():
    global _connected
    if _client is not None:
        try:
            _client.disconnect()
        except Exception as exc:
            print(f"[score-mqtt] Error while disconnecting: {exc}")
    _connected = False


def publish_score(score, misses):
    """Publishes the current score/misses. Returns True if actually
    published, False if skipped (not connected) -- never raises, so a
    dropped connection mid-game can't crash the game loop."""
    if not _connected:
        return False
    try:
        _client.publish(SCORE_TOPIC, f"score={score} misses={misses}")
        return True
    except Exception as exc:
        print(f"[score-mqtt] Publish failed ({exc}) -- continuing without it this time.")
        return False
