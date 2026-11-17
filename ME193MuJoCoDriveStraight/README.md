# Drive straight: a two-beam walker from CAD to MuJoCo to a LEGO car

A LEGO Double Motor turns two beams instead of wheels. We modelled the robot in Onshape, exported it to MuJoCo,
trained a policy with [mjlab](https://github.com/mujocolab/mjlab) (PPO) to walk in a straight line, and ran that
policy on the real LEGO car over Bluetooth.

| | |
|---|---|
| CAD | Onshape assembly `sillywalk` (link in `onshape/config.json`), exported with onshape-to-robot 1.8.3 |
| Simulator | MuJoCo 3, mjlab 1.6.0, trained on a Tufts HPC GPU |
| Real robot | LEGO Double Motor (built-in IMU, two leg encoders), Connection Card orange 0994, laptop over Bluetooth |
| Result | The trained policy holds heading on the real car (best runs: finished within 3° of straight, 6° average error); with no policy the car turns up to 115° in 10 s |

## What is in this folder

| Path | What it is |
|---|---|
| `onshape/` | The exported robot: `robot.xml` (bodies, leg joints, motors), `scene.xml` (robot + floor), `assets/` (STL meshes) |
| `drive.py` | Scripted drive test in plain MuJoCo (sit, walk, spin, curve), films `drive.mp4` |
| `mjlab_task/drive_task.py` | The RL tasks `Drive-Straight` and `Drive-Straight-Lego`: observations, actions, rewards, randomization |
| `mjlab_task/train.py`, `make_video.py`, `play.py`, `train.sbatch` | Train, film and play back a policy (run on the cluster) |
| `docs/how-drive-straight-works.md` | The training idea in plain words |
| `lego/` | Everything for the real car: measure it, export the policy to numpy, run it, fit the sim drift (own README) |
| `results/videos/` | Simulation videos (below) |
| `results/checkpoints/` | Trained policies (`.pt`) |
| `results/lego_logs/` | One CSV per real-car run (time, heading, turn rate, leg commands) |
| `results/training_params/` | The exact PPO and environment settings of the final policy |

## From CAD to simulation

1. The Onshape assembly has a body and two beams, each joined to the body by a revolute mate named `dof_left_leg` /
   `dof_right_leg`; onshape-to-robot turns those into hinge joints `left_leg` and `right_leg`.
2. The right mate was exported as `dof_right_leg_inv`, which reversed that leg: equal commands made the beams turn
   against each other and the robot spun in place. The right joint's axis is flipped in `robot.xml`
   (same as renaming the mate to `dof_right_leg`).
3. The beam axles point along x, so this robot walks along **+y**. The tasks use `FWD, SIDE = 1, 0` for that.
4. In plain MuJoCo, both legs at 8 rad/s walk about 0.08 m/s in a straight line (`results/videos/sim_scripted_drive_test.mp4`).

## The policy

A small neural network (MLP 64 x 64, ELU) trained with PPO (rsl_rl via mjlab): 4096 simulated robots, 1000 iterations.
Every 0.1 s it reads the robot's state and outputs a target speed for each leg.

**Inputs (`Drive-Straight-Lego`, 5 numbers):** sin(heading), cos(heading), turn rate, and its previous two outputs.
These are only things the real car can measure: heading and turn rate come from the Double Motor's IMU.
The first task, `Drive-Straight`, also used body velocity and distance from the line, which the car cannot measure,
so its policy could not be moved to the car.

**Outputs:** one speed per leg, -1..1, scaled to 12 rad/s (73% LEGO speed).

**Reward (every step):**

| Term | Formula | Weight | Why |
|---|---|---|---|
| `forward_speed` | 1 - ((v_forward - 0.08) / 0.08)^2, floored at -1 | +1.0 | walk forward at 0.08 m/s; standing still earns 0, backwards is punished |
| `face_forward` | -(heading error)^2 | +1.0 | keep pointing forward (the one thing the car can see) |
| `stay_on_line` | -(sideways offset)^2 | +0.5 | do not drift sideways |
| `action_rate` | squared change of the leg commands | -0.05 | smooth commands, kind to real motors |

An episode ends after 10 s, if the robot tips past 60°, or wanders 1.5 m off the line. Each robot starts turned up to
±23° off forward, so it has to learn to steer back.

**Made robust for the real car:** noise on the heading and turn-rate inputs, the turn rate up to one step late
(Bluetooth), and per-robot random leg friction and motor strength. The training critic sees the full state
(position, velocities) while the policy only sees the 5 inputs.

## Sim to real

Measured on the car with `lego/hw_calibrate.py`: leg speed is linear, 0.165 rad/s per 1% (100% = 15.9 rad/s), both
legs equal; the read + command loop runs at 15.9 Hz, so the policy runs at 10 Hz; the hub's yaw is in 0.1° and
counter-clockwise positive, the same as MuJoCo.

`lego/export_policy.py` copies the network into `policy_lego.npz` (checked against torch: max difference 8e-7), and
`lego/lego_run.py` runs it with numpy on the laptop and sends `movement_move_tank(left%, right%)` to the car.

### Simulation videos (`results/videos/`)

| Video | What it shows |
|---|---|
| `sim_trained_policy_lego.mp4` | Trained policy, 10 s: 0.74 m forward, 4 cm sideways drift, heading within ±11° |
| `sim_no_policy_symmetric.mp4` | No policy, both legs 50%: the perfectly symmetric sim robot walks straight (heading 2°) |
| `sim_no_policy_fitted_drift.mp4` | No policy with a drift fitted to the real car: follows the real heading within 5.4° RMS |
| `sim_first_policy_13_inputs.mp4` | First policy (13 inputs, not usable on the car): 0.79 m, but 18 cm sideways and ~14° off |
| `sim_scripted_drive_test.mp4` | Scripted walk, spin and curve |

### Real car (10 s runs, `results/lego_logs/`)

| Run | Final heading | Average heading error | Worst | Time with a leg at the 73% limit |
|---|---|---|---|---|
| No policy, 15 Nov 12:11 | -115.3° | 59.9° | 115.9° | – |
| **Policy, 15 Nov 12:12** | **+0.2°** | **5.9°** | 24.4° | 4.4 s |
| No policy, 15 Nov 12:18 | +26.6° | 7.5° | 26.6° | – |
| **Policy, 15 Nov 12:19** | **+2.5°** | **6.4°** | 25.8° | 5.2 s |
| No policy, 15 Nov 12:20 | -86.9° | 41.2° | 91.6° | – |
| Policy, 15 Nov 12:20 | -19.7° | 17.8° | 37.5° | 8.3 s |
| Policy, 17 Nov 10:37 | -18.1° | 22.3° | 56.1° | 8.3 s |
| Policy, 17 Nov 10:38 | +17.3° | 9.8° | 33.8° | 6.2 s |
| **Policy, 17 Nov 10:38** | **+7.8°** | **7.1°** | 19.7° | 4.9 s |
| **Policy, 17 Nov 10:39** | **+2.3°** | **8.5°** | 33.7° | 5.3 s |
| Policy, 17 Nov 10:39 | -20.0° | 10.6° | 29.7° | 6.5 s |

All CSVs are in `results/lego_logs/`, including a few more 15 Nov policy runs (one started while the car was
being handled and spun). The 15 Nov runs connected to the first Double Motor found; since 17 Nov the scripts only
connect to card orange 0994.

### What we learned

- **The real car is not symmetric; the simulated one is.** With no policy the sim robot walks straight, while the real
  car turned anywhere from +27° to -115° in 10 s. The drift changes from run to run (start angle of the beams, floor).
- **Grip differences between the legs do not explain it** in simulation (at most a few degrees, wrong direction), nor
  do legs starting out of phase (up to ~40°, not a steady turn). One leg doing ~32% more effective work per turn
  reproduces the real heading curve (`lego/sim_openloop_drift.py`). The legs spin at the same rate, so this is an
  effective stride difference standing in for the car's unevenness, not a measured cause.
- **MuJoCo uses the larger friction of two touching surfaces**, and the floor's is 1, so the leg-friction
  randomization in training (0.4 to 1.2) had almost no effect.
- **The policy steers correctly but runs out of authority**: in most real runs one leg sits at the 73% limit for 5 to
  8 of the 10 s. When the car pulls harder than that, the heading settles 15 to 20° off.
- **Next step:** raise the limit to ~95% and randomize a per-leg stride difference of ±35% during training, so the
  policy practises on lopsided robots before meeting the real one.

## How to run

**Simulation** (laptop, `~/.venvs/onshape` has mujoco + imageio):
```
MUJOCO_GL=cgl python drive.py onshape/scene.xml
```

**Training** runs on a GPU with the course's `mujoco-hpc` toolkit (`runIt.py`), which copies the task and the robot
to the cluster. In that toolkit this folder's contents live under `09_onshape_robot/`:
```
python3 runIt.py --gpu --time 00:30:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
  09_onshape_robot/mjlab_task/train.py Drive-Straight-Lego --env.scene.num-envs 4096
python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
  --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape --with <model_999.pt> \
  09_onshape_robot/mjlab_task/make_video.py model_999.pt Drive-Straight-Lego
```

**Real car** (from `lego/`, hub with card orange 0994 awake, car flat on the floor pointing straight):
```
~/.venvs/me193/bin/python lego_run.py --open-loop   # no policy, both legs 50%
~/.venvs/me193/bin/python lego_run.py               # trained policy
```
`~/.venvs/me193` needs `legoeducation==1.2.0` and numpy. To export a new checkpoint:
`python export_policy.py <model_n.pt>` (needs torch).
