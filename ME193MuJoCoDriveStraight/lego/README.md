# Running the trained policy on the real LEGO car

The car is a LEGO Double Motor (two beams for legs, built-in IMU) plus a color sensor, controlled from the laptop
over Bluetooth with `lelib.py` (copied from `../../ME193Sumo/`). It only connects to the hub with Connection Card **orange 0994**. Python: `~/.venvs/me193` (car), `~/.venvs/me193-torch` (export).

| Step | Command (from this folder) | What it gives |
|---|---|---|
| 1. Measure the car | `~/.venvs/me193/bin/python hw_calibrate.py` | `lego_calibration.json`: 0.165 rad/s per 1% speed (100% = 15.9 rad/s), 15.9 Hz Bluetooth loop |
| 2. Train in sim | `Drive-Straight-Lego` task (see below) | `model_<n>.pt` |
| 3. Export | `~/.venvs/me193-torch/bin/python export_policy.py <model_<n>.pt>` | `policy_lego.npz` (numpy weights) |
| 4. Baseline | `~/.venvs/me193/bin/python lego_run.py --open-loop` | how much the car curves with no policy |
| 5. Policy | `~/.venvs/me193/bin/python lego_run.py` | the policy driving the car; CSV log in `../results/lego_logs/` |

Measured facts the sim task is built on (2026-11-15):
- hub yaw is in 0.1 deg, wraps at +-180 deg, counter-clockwise positive (same as MuJoCo)
- both legs reach the same speed for the same %; the car still curves 20-35 deg in 3 s from the beams slipping
- `movement_move_tank(+, +)` walks forward; the motors are mirrored inside the hub (it reports -29/+29)

## Train (laptop, `mujoco-hpc` folder, VPN on)

```
python3 runIt.py --gpu --time 00:30:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  09_onshape_robot/mjlab_task/train.py Drive-Straight-Lego --env.scene.num-envs 4096
```

The policy only sees what the car measures: sin/cos of heading, yaw rate, and its last action (5 numbers, 10 Hz).
Leg friction and motor strength are randomized per robot so it copes with the real car's curving.
Film it with `make_video.py model_<n>.pt Drive-Straight-Lego`.
