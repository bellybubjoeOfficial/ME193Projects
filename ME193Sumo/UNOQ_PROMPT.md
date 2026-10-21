You are working on the **Arduino UNO Q** side of a robotics class project (Tufts ME193 "sumo wrestler"). Another Claude session runs on the team's laptop and owns the robot's decision-making code. Your job is to turn the UNO Q into a reliable **sensor node that publishes over MQTT** in exactly the format below, so the laptop can act on it.

## The system (what already exists, on the laptop, not your job)

- Two LEGO robots fight in a ring of blue tape; pushed out = lose.
- The **laptop** is the brain (Python, repo folder `ME193Sumo/`). It runs a hand-written **decision tree** every 30 Hz tick. It is already connected to and controls: an Xbox controller (teleop), a LEGO Double Motor (two crank "feet"), a LEGO Single Motor (a whacker arm on RT/LT), and a downward LEGO color sensor (blue tape → beep + controller rumble + reverse blocked for 1.5 s).
- The **UNO Q** is mounted on the robot. It adds the sensors the LEGO kit doesn't have: first a **distance sensor** (to tell how close the opponent is), maybe an **IMU** later (the user will specify).
- Next step on the laptop (after your part works): the distance reading will gate the whacker. The robot will refuse to swing when the opponent is far away, swing on its own when the driver holds an "armed" button and the opponent gets close, and lock the whacker if your data goes stale (> 0.3 s old). So **rate, latency and honesty ("unknown" ≠ "far") matter more than anything else**.
- The project is built one tested stage at a time: every piece gets unit tests plus a short hardware check with a written pass criterion. Please work the same way.

## Board / network facts

- UNO Q hostname `UNOQ3`, IP `10.247.137.61`, user `arduino`. Ask the user for the sudo password; don't write it into any file.
- The laptop (`10.243.98.48`) is on a **different campus subnet**. Laptop → Q is known to work (SSH, ping). Q → laptop is unproven. That's why **the MQTT broker runs on the Q**.
- `.local` names don't resolve on this network; use IPs.

## MQTT contract (must match exactly; the laptop already parses and validates this)

**Broker:** Mosquitto **on the UNO Q**, port **1883**, listening on **all interfaces** (`listener 1883 0.0.0.0`), `allow_anonymous true` (class network only). Your publisher connects to `localhost:1883`. The laptop connects to `10.247.137.61:1883`.

All payloads are UTF-8 **JSON objects**. Topic prefix `me193sumo/`.

| Topic | Direction | Rate | QoS | Retain |
|---|---|---|---|---|
| `me193sumo/unoq/status` | Q → laptop | on change | 1 | yes |
| `me193sumo/unoq/distance` | Q → laptop | **20 Hz target, ≥ 15 Hz required** | 0 | no |
| `me193sumo/unoq/imu` | Q → laptop | 50 Hz target (later, only when asked) | 0 | no |
| `me193sumo/laptop/ping` | laptop → Q | on demand | 0 | no |
| `me193sumo/unoq/pong` | Q → laptop | echo of each ping | 0 | no |

- **status**: on connect publish `{"state": "online", "sensors": ["distance"], "version": "0.1.0"}` (retained, QoS 1). Set an MQTT **Last Will** of `{"state": "offline"}` (retained, QoS 1) so the laptop knows immediately if you die.
- **distance**: `{"seq": 1532, "t": 1760891234.512, "mm": 423}`
  - `seq`: int, +1 per message, starts at 0 when the publisher starts (the laptop counts gaps as lost messages and handles restarts).
  - `t`: `time.time()` on the Q when the reading was taken (float seconds, logging only; the clocks aren't synced).
  - `mm`: distance in **millimetres as an integer**, or **`null`** when the sensor has no valid target / is out of range / the read failed. **Never** send a fake large number for "nothing there", and never send floats, strings or negatives. The laptop treats `null` as "unknown", which is the safe behavior.
- **imu** (don't build until asked): `{"seq", "t", "ax", "ay", "az", "gx", "gy", "gz"}`, accel in g, gyro in deg/s, board frame.
- **ping/pong**: whenever a message arrives on `me193sumo/laptop/ping`, immediately republish **the exact same payload, unchanged** to `me193sumo/unoq/pong`. The laptop measures round-trip latency with it.
- Adding a sensor later = a new topic `me193sumo/unoq/<name>` with `seq` + `t` + its fields, plus its name in `status.sensors`. Don't change existing fields without telling the user, because the laptop validates them strictly.

## What to build

1. **Mosquitto on the Q**: install, configure as above, enabled at boot. Verify locally with `mosquitto_sub`/`mosquitto_pub`.
2. **Find out how the distance sensor is actually connected before writing code.** Ask the user which part it is and where it's plugged in. On the UNO Q the Qwiic connector is typically wired to the **STM32 microcontroller**, not the Linux side. If so, the reading has to come from an MCU sketch and cross to Linux through the Arduino Bridge (RPC) that App Lab apps use. Check the installed App Lab / Bridge docs and examples on the board for the real API rather than guessing. If the sensor is instead on an I2C bus visible to Linux (`/dev/i2c-*`), reading it directly from Python is simpler. Pick the simplest path that reliably hits ≥ 15 Hz and explain the choice.
3. **Publisher (Python on the Linux side)**: reads distance at ~20 Hz, publishes per the contract, echoes pings, sets the Last Will, **reconnects automatically** if the broker restarts, and **never crashes on a bad sensor read** (publish `mm: null` and keep going). Keep the timing loop drift-free so the rate holds.
4. **Starts on boot** (systemd service, or the App Lab app's autostart, whichever fits step 2), so the robot works in a match without anyone SSHing in.
5. **Tests**:
   - Unit tests (no hardware) for the payload builder: types, `null` handling, `seq` increments, and the pong echo returning the payload unchanged.
   - A local hardware check script that subscribes on the Q itself and confirms rate ≥ 15 Hz and valid JSON.
   - Tell the user how to stop the publisher (to test the Last Will) and how to view its logs.

## Organisation

Keep everything in one folder on the Q, e.g. `~/me193sumo_unoq/`, as a git repo so it can be copied into the team's GitHub repo later (`ME193Sumo/unoq/`):

```
me193sumo_unoq/
  README.md            how it works, how to run / stop / view logs, pinout
  MQTT_CONTRACT.md     copy of the contract above
  publisher/           Python publisher + payload helpers
  sketch/              MCU sketch (only if the sensor is on the MCU)
  tests/               unit tests + hw check
  deploy/              mosquitto conf snippet, systemd unit(s), install steps
```

Use a Python virtualenv (or the board's packaged `python3-paho-mqtt`) rather than `sudo pip`. Match MQTT library major version 2 (`paho-mqtt>=2.0`, `CallbackAPIVersion.VERSION2`) if you use paho.

## Definition of done

On the **laptop**, the user runs (from `ME193Sumo/`):

```
~/.venvs/me193/bin/python tests/hw_unoq.py --live
```

and it prints **ALL PASS**: broker reachable, status `online`, distance ≥ 15 Hz, 0 malformed messages, < 5% missed (seq gaps), median ping round trip < 100 ms. Then stopping your publisher makes the laptop show distance `---` within 0.3 s and status `offline`.

When finished, report back in this format so the user can paste it to the laptop Claude:

- Sensor part + how it's connected (MCU via Bridge, or Linux I2C)
- Measured publish rate and anything that limits it
- Sensor range (min/max mm) and what it reports when nothing is in front
- Any deviation from the contract (ideally none)
- How to start / stop / restart / view logs
- Where the code lives
