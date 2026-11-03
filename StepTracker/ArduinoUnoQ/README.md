# Shin IMU streamer (Arduino UNO Q)

The device side of the shin-mounted step counter. Its only job is to read the
IMU at a steady 100 Hz, timestamp every sample, and stream the samples to the
laptop over MQTT. All ML runs on the laptop.

```
MPU-6050 ──I2C──▶ STM32 MCU ──Router Bridge──▶ Linux (Python) ──MQTT/Wi-Fi──▶ laptop
                  sketch/                       streamer/                       Mosquitto
                  100 Hz timer, stamps,         clock map, seq, units,          + PyTorch
                  batches of 10                 ring buffer, QoS 1, sync
```

## How it works

**MCU (`sketch/sketch.ino`)**
- A Zephyr `k_timer` fires every 10 ms. A high-priority sampler thread wakes on
  it, reads the MCU's 64-bit cycle counter (ns), then burst-reads the 14 data
  bytes over I2C. No `delay()`-paced loop.
- Each sample's MCU seq is the index of its timer tick. If a tick is ever
  missed (I2C error, queue full), the seq skips, so a loss shows up as a gap
  and is never silent.
- The IMU is set to ±16 g and ±2000 °/s (the maximum ranges). Both registers
  are read back, and the streamer reports what the chip actually has. The DLPF
  is set to 44 Hz at 100 Hz sampling, which stays under Nyquist.
- `loop()` packs 10 samples (24 bytes each) into one `Bridge.notify("imu_batch")`,
  which is 10 messages/s over the Bridge UART. Sending never delays a read
  because the sampler thread preempts `loop()`.

**Why the Bridge rather than a raw serial port:** on the UNO Q the MCU's UART
to Linux *is* the Bridge, owned by `arduino-router`. The Bridge gives framing
(MessagePack) and RPC for free. At 10 batches/s it uses roughly 25 % of the
link, which leaves room for clock-sync calls.

**Linux (`streamer/`, plain Python under systemd, not an App Lab app)**
- `router.py` speaks the arduino-router MessagePack-RPC protocol directly on
  `/var/run/arduino-router.sock` (the same protocol App Lab's `Bridge` uses).
- `clock.py` maps MCU time to Linux `CLOCK_MONOTONIC`. Once a second it calls
  `now_ns()` on the MCU and brackets the call with Linux timestamps. It keeps
  the lowest-RTT point per 10 s and fits offset + drift over 10 min.
  `t_mono_ns` is the read time mapped to Linux. `t_wall_ns` = `t_mono_ns` +
  (`CLOCK_REALTIME` − `CLOCK_MONOTONIC`), so it is the same wall clock that
  the sync pongs use.
  This board's MCU clock was measured at **~880 ppm slow** (internal
  oscillator), which is about 0.5 s per 10 min, so the drift term matters.
  It also means the true sample rate is ~99.91 Hz, not exactly 100. Use the
  timestamps, not `seq × 10 ms`, when you resample.
- `seq` starts at 0 for each session and never goes backwards. It survives MCU
  seq wrap and even an MCU reboot.
- `ring.py` buffers samples while the broker is unreachable. The default is
  300 s; the requirement is ≥ 60 s. If the buffer ever fills, the **oldest**
  samples are dropped, and every drop is logged and counted
  (`ring_dropped` in `wearable/diag`).
- Publishing uses QoS 1 with at most 20 unacknowledged messages. Unacked
  messages are resent on reconnect, and everything else is released from the
  ring in order. paho-mqtt v2 handles the auto-reconnect (1–8 s backoff,
  keepalive 5 s).

## Repo layout

```
device/
├── config/streamer.toml     broker host/port and all settings  ← edit this
├── sketch/                  MCU firmware (sketch.ino, sketch.yaml)
├── streamer/                Linux publisher (python -m streamer --config …)
├── systemd/shin-imu.service user unit
├── scripts/flash_sketch.sh  compile + flash the MCU from the board
├── scripts/install.sh       venv + install/enable the service
├── tools/                   test tools: record, analyze, sync_test, netcut
└── tests/                   pytest: units + end-to-end with a fake MCU
```

## Setup

All of these steps run **on the board** (`ssh arduino@<board-ip>`).

1. **Wiring.** Connect the MPU-6050 to the **SDA/SCL header pins** (next to
   AREF; this is the `Wire` bus). On the UNO Q these pins are *not* the same
   as A4/A5. The sketch scans `Wire`, then `Wire2` (A4/A5), then `Wire1`
   (Qwiic), at addresses 0x68/0x69.

2. **Free the MCU from App Lab.** An App Lab app flashes its own sketch when
   it starts, including at boot if it is the default app:
   ```bash
   arduino-app-cli app list                          # anything "running"?
   arduino-app-cli app stop ~/ArduinoApps/<that-app>
   arduino-app-cli properties set default none
   ```

3. **Flash the sketch.** This takes about 2 min to compile and 10 s to flash:
   ```bash
   cd ~/shin-step-counter/device
   scripts/flash_sketch.sh
   ```

4. **Configure the broker** in `config/streamer.toml`, under `[mqtt] host`
   and `port`. The host is the laptop's IP on the **same network as the
   board**; check it with `ipconfig` or `ip addr` on the laptop. On the laptop,
   Mosquitto 2.x needs to listen beyond localhost:
   ```
   # mosquitto.conf
   listener 1883 0.0.0.0
   allow_anonymous true        # or set up a user; the password goes in ~/.config/shin-imu/env
   ```
   If you use a password, set `username` in the TOML and put
   `SHIN_IMU_MQTT_PASSWORD=...` in `~/.config/shin-imu/env`, not in the repo.

5. **Install the service.** It starts at boot and restarts on crash:
   ```bash
   scripts/install.sh
   systemctl --user status shin-imu
   journalctl --user -u shin-imu -f     # summary line every 10 s
   ```
   After editing the config: `systemctl --user restart shin-imu`.

**LED4** shows the state: green means streaming and connected, blue means
streaming but buffering (no broker), red means no IMU found, and off means
stopped.

## MQTT topics

| Topic | Dir | QoS | Payload |
|---|---|---|---|
| `wearable/imu/raw` | out | 1 | `{"device","session","rate_hz","fields","units","samples":[[seq,t_mono_ns,t_wall_ns,ax,ay,az,gx,gy,gz],…]}`, 10 samples per message |
| `wearable/status` | out | 1, retained | every 1 s: `{"device","online","session","rate_hz_measured","samples_sent","samples_buffered","reconnects","accel_range_g","gyro_range_dps","uptime_s"}`. The Last Will publishes the same shape with `online:false`. |
| `wearable/diag` | out | 0, retained | every 5 s: MCU counters (late, i2c_err, overflow), seq gaps, ring drops, peaks, saturation count, clock-fit info |
| `wearable/sync/ping` | in | 0 | `{"id","t_laptop_ns"}` |
| `wearable/sync/pong` | out | 0 | `{"id","t_laptop_ns","t_dev_recv_ns","t_dev_send_ns"}` (device wall clock, ns) |
| `wearable/cmd` | in | 1 | `{"cmd":"start"}`, `{"cmd":"stop"}`, `{"cmd":"set_rate","value":25\|50\|100\|200}` |
| `wearable/cmd/ack` | out | 1 | `{"cmd","ok","detail"}` |

Notes for the laptop side:
- `samples_sent` counts samples whose PUBACK came back. `samples_buffered`
  counts everything not yet acknowledged.
- `start` creates a new session UUID, so seq restarts at 0 there (and only
  there). The service autostarts a session at boot (`autostart` in the config).
  A service restart, such as after a crash, also starts a new session. Samples
  held in RAM at the moment of a crash are lost.
- `set_rate` keeps the session. Messages after the change carry the new
  `rate_hz`.
- Sync pongs use QoS 0 on purpose: a QoS 1 pong could queue behind a backlog
  flush, and then its `t_dev_send_ns` would no longer be when it left.
- Example: `mosquitto_pub -h <laptop> -t wearable/cmd -m '{"cmd":"set_rate","value":50}'`

## Running the tests

The unit tests and end-to-end tests run on the board with no hardware. The
end-to-end tests drive a fake MCU into a real `mosquitto` process:
```bash
~/.venvs/shin-imu/bin/python -m pytest
```

For the acceptance tests on hardware, `tools/record.py` subscribes to
everything and writes JSON lines, and `tools/analyze.py` checks rate, gaps,
jitter and peaks. Run the recorder on the laptop, or on the board against
its local Mosquitto. Use `--host` for the broker.

```bash
PY=~/.venvs/shin-imu/bin/python   # on the laptop: any Python with paho-mqtt 2.x

# 1. Bench, 10 min: rate within ±1 %, no seq gaps, interval std reported
$PY tools/record.py --host <broker> --seconds 600 --out bench.jsonl
$PY tools/analyze.py bench.jsonl

# 2. Disconnect, 30 s: either turn off the laptop's Wi-Fi / stop Mosquitto for 30 s
#    while recording, or use the proxy that black-holes the link (no FIN/RST,
#    like a dead radio). Point a streamer at the proxy, then:
$PY tools/netcut.py --listen 18831 --to 127.0.0.1:1883 --cut-after 60 --cut-for 30 &
$PY tools/record.py --host <broker> --seconds 150 --out disc.jsonl
$PY tools/analyze.py disc.jsonl      # expect 0 gaps, status_buffered_max ≈ 3000+

# 3. Range: record while walking / stomping / tapping the IMU hard
$PY tools/record.py --host <broker> --seconds 60 --out range.jsonl
$PY tools/analyze.py range.jsonl     # peak_abs, peak_accel_norm_g, samples_at_*_full_scale

# 4. Sync: 20 pings (run on the laptop: that's the clock being synced to)
$PY tools/sync_test.py --host <broker> --n 20
```

`tools/record.py` uses a persistent MQTT session, so messages published
while the recorder itself was reconnecting are still delivered. For that to
work when Mosquitto is restarted, the broker needs `persistence true`.

## Troubleshooting

- **Log says "the MCU is not running the shin_imu sketch".** An App Lab app
  re-flashed the MCU. Repeat setup step 2, then run `scripts/flash_sketch.sh`.
- **LED is red, or `wearable/diag` shows `"source": null`.** No IMU was found.
  Check the wiring on the SDA/SCL header. The MCU rescans every 2 s.
- **Log says "cannot reach MQTT broker".** Check the `host` in the config, that
  the laptop is on the same network, and Mosquitto's `listener` line. Samples
  keep buffering in the meantime.
- **Switching back to the activity classifier:**
  ```bash
  systemctl --user disable --now shin-imu
  arduino-app-cli app start ~/ArduinoApps/activity_wearable
  ```
  That app looks for the IMU on A4/A5, so it needs that wiring.
