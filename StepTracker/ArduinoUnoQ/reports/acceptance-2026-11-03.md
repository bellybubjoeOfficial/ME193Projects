# Acceptance tests — 2026-11-03

Board: UNO Q (`arduino,imola`), sketch `shin_imu/1`, IMU MPU-6050 on `Wire`
(SDA/SCL header) at 0x68, WHO_AM_I 0x68. Ranges read back from the chip:
ACCEL_CONFIG 0x18 = ±16 g, GYRO_CONFIG 0x18 = ±2000 °/s, DLPF 0x03 (44 Hz).

Broker: the board's own Mosquitto (127.0.0.1). The laptop broker in
`config/streamer.toml` was unreachable, because the board is now on a
different subnet. These tests therefore cover MCU → Bridge → streamer →
MQTT, but not the Wi-Fi hop.

## 1. Bench, 10 min — PASS

| | |
|---|---|
| Duration / samples | 622.8 s / 62,230 (seq 0–62229) |
| Measured rate | 99.912 Hz (−0.088 %) |
| Seq gaps / duplicates / out of order | 0 / 0 / 0 |
| Messages | 6,223, all exactly 10 samples |
| Interval (Δt_mono), mean / std | 10,008.8 µs / **7.9 µs** |
| Interval p1 / p50 / p99, min / max | 9,996.8 / 10,008.8 / 10,058.8 µs, 9,950 / 10,068 µs |
| MCU late / I2C errors / queue overflow | 0 / 0 / 0 |
| Median latency, read → laptop receive | 30 ms (a 10-sample batch spans 100 ms) |

The rate is 0.088 % low because the MCU clock runs ~880 ppm slow (measured
independently, then tracked by the clock fit: 880–895 ppm). The timer period
is exact in MCU time. Resample on the timestamps.

Fixes made while testing:
- The clock fit clamped drift at 500 ppm, below the real 880 ppm. The limit
  is now 10,000 ppm.
- The MCU→Linux mapping stepped by up to 22 ms at each refit, which showed
  up as interval std 112.6 µs and min/max intervals of 2.8 / 32 ms. It now
  slews by at most 0.5 % of each interval, giving std 7.9 µs.

## 2. Disconnect, 30 s — PASS

`tools/netcut.py` black-holed the MQTT TCP link for 30 s (no FIN/RST, like a
lost radio link).

| | |
|---|---|
| Link down → detected | 8 s (keepalive 5 s) |
| Link up → reconnected | 8 s (reconnect backoff) |
| Peak `samples_buffered` | 2,970 |
| Delivered | 14,930 / 14,930, seq 0–14929, 0 gaps, 0 duplicates |
| Rate / interval std | 99.912 Hz / 12.6 µs |
| Final status | `online:false`, `samples_buffered:0` (clean shutdown) |

The Last Will on an unclean exit is covered by
`tests/test_end_to_end.py::test_last_will_on_crash`. A `kill -9` of the
service was also tried: systemd restarted it within 2 s (`NRestarts=1`).

## 3. Range — PASS (no clipping)

60 s of shaking, tapping and stomping by hand: 6,010 samples, 0 gaps.

| | Peak | Full scale |
|---|---|---|
| Accel (axis) | 25.1 m/s² = 2.56 g (ax), 24.8 (az), 19.5 (ay) | ±16 g |
| Accel (norm) | 3.64 g | |
| Gyro | 417 °/s (gy), 283 (gx), 92 (gz) | ±2000 °/s |
| Samples at full scale | 0 accel, 0 gyro | |

At rest, the gyro bias is about 2 °/s on z, so subtract it on the laptop if
it matters.

## 4. Sync, 20 pings — PASS

Pings were sent while streaming at 100 Hz. The pinger ran on the board, so
this RTT excludes Wi-Fi; expect it to grow by the Wi-Fi round trip when run
from the laptop.

| | |
|---|---|
| Pongs | 20 / 20 |
| RTT min / median / mean / max | 2.67 / 4.97 / 7.11 / 45.5 ms |
| Device hold (t_dev_send − t_dev_recv) | 130 µs median |
| Offset at min RTT | −0.19 ms (same host, so ≈ 0 as expected) |

Use the lowest-RTT pongs for the offset estimate. The max RTT comes from
pings that queued behind data messages in the broker.
