# MQTT contract — UNO Q ⇄ laptop

Single source of truth for the messages between the Arduino UNO Q (sensors)
and the laptop (decision tree, `ME193Sumo/`). Change it here first, then
on both sides.

## Broker

- **Mosquitto runs on the UNO Q**, port **1883**, listening on all
  interfaces, anonymous access allowed (class network only).
- Laptop connects to `10.247.137.61:1883` (`config.MQTT_BROKER`).
- The Q's own publisher connects to `localhost:1883`.

Why on the Q: the laptop and the Q are on different campus subnets. We know
laptop → Q works (SSH), and with the broker on the Q that's the only path
needed.

## Topics (prefix `me193sumo/`)

| Topic | Direction | Rate | QoS | Retain |
|---|---|---|---|---|
| `me193sumo/unoq/status` | Q → laptop | on change | 1 | yes |
| `me193sumo/unoq/distance` | Q → laptop | 20 Hz target (≥ 15 Hz) | 0 | no |
| `me193sumo/unoq/imu` | Q → laptop | 50 Hz target (later) | 0 | no |
| `me193sumo/laptop/ping` | laptop → Q | on demand | 0 | no |
| `me193sumo/unoq/pong` | Q → laptop | echo of each ping | 0 | no |

All payloads are UTF-8 JSON objects.

### `status`
```json
{"state": "online", "sensors": ["distance"], "version": "0.1.0"}
```
- Publish `online` (retained) on connect.
- Set an MQTT **Last Will** of `{"state": "offline"}` (retained, QoS 1) so
  the laptop learns immediately if the Q drops off.

### `distance`
```json
{"seq": 1532, "t": 1760891234.512, "mm": 423}
```
- `seq`: integer, +1 per message, starts at 0 when the publisher starts.
- `t`: Q's `time.time()` when the reading was taken (seconds, float).
  Logging only; the clocks aren't synced, so the laptop measures staleness
  with its own receive time.
- `mm`: distance in **millimetres** (integer), or `null` when the sensor
  sees nothing in range / the reading is invalid.

### `imu` (reserved, not built yet)
```json
{"seq": 88, "t": 1760891234.512, "ax": 0.01, "ay": -0.02, "az": 1.00, "gx": 0.5, "gy": -0.1, "gz": 12.0}
```
- accel in **g**, gyro in **deg/s**, axes in the board frame (document
  the mounting on the robot once it's fixed).

### `ping` / `pong` (round-trip latency)
- Laptop publishes `{"id": 7, "t": <laptop time>}` to `me193sumo/laptop/ping`.
- Q immediately republishes **the same payload unchanged** to
  `me193sumo/unoq/pong`. The laptop computes the round trip on its own clock.

## Laptop-side rules

- A sensor counts as **stale** if no valid message has arrived for
  `SENSOR_STALE_S` = 0.3 s. Stale or `mm: null` means "unknown", never "far away".
- Malformed messages (bad JSON, missing `seq`, wrong types) are counted
  and dropped, never crash the loop.
- Adding a sensor = new topic `me193sumo/unoq/<name>` with `seq` + `t` +
  its fields, added to this file, the `status.sensors` list, and
  `unoq_link.SENSORS` on the laptop.
