# ME193 Sumo Wrestler

Tufts ME193 robot sumo: a LEGO Education robot on crank feet with a swinging whacker, driven with an Xbox controller. An Arduino UNO Q on the robot sends distance readings over MQTT. A hand-written **decision tree** (`policy.py`) runs 30 times a second and combines the driver's input with the sensors (shared control).

## Hardware
- **LEGO Double Motor**: two crank feet. When the stick is released, each foot settles into the center of its calibrated base position.
- **LEGO Single Motor**: the whacker. It rests at base, strikes across when an opponent comes within 200 mm, then returns. RT/LT override it within the same end-point limits.
- **LEGO Color Sensor** (facing down): seeing the blue ring tape triggers a beep and a controller rumble, and blocks reverse for 1.5 s.
- **Arduino UNO Q**: distance sensor, published over MQTT (see `MQTT_CONTRACT.md`).
- **Xbox controller**: left stick drives, B brakes, RT/LT swing the whacker, Y plays the sad tune, A plays the victory tune.

## Run
```bash
python -m pip install -r requirements.txt
python main.py                 # everything
python main.py --no-unoq       # without the UNO Q
python -m unittest discover -s tests -t . -v   # logic tests (no hardware)
```
Every tunable value is in `config.py`. Calibrations: `tests/hw_feet_calibrate.py` (feet base range) and `tests/hw_whacker_calibrate.py` (whacker positions where it blocks the distance sensor).

## Docs
- `policy.py`: the decision tree, drawn in its docstring
- `TEST_PLAN.md`: build stages, each with logic tests and hardware checks
- `MQTT_CONTRACT.md`: message format between the UNO Q and the laptop
