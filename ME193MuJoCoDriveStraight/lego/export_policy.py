"""
Turn a trained Drive-Straight-Lego checkpoint (model_<n>.pt) into policy_lego.npz, which lego_run.py runs with
plain numpy on the laptop (no torch, no mjlab).

Usage (laptop):
    ~/.venvs/me193-torch/bin/python export_policy.py <path to model_<n>.pt>

The actor is: normalize the 5 inputs -> Linear 64 -> ELU -> Linear 64 -> ELU -> Linear 2. At run time we use its mean
output (no exploration noise). The two outputs are in the order the leg joints appear in robot.xml (mjlab keeps
the model's joint order), which this script reads and stores so lego_run.py sends each number to the right leg.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROBOT_XML = HERE.parent / "onshape" / "robot.xml"
EPS = 1e-2   # rsl_rl EmpiricalNormalization: (x - mean) / (std + 1e-2)

# must match drive_task.py (Drive-Straight-Lego) and the measured car
LEGO_MAX_LEG_SPEED = 12.0     # rad/s at action 1.0
LEGO_CONTROL_HZ = 10
OBS_NAMES = ["sin_yaw", "cos_yaw", "yaw_rate", "last_action_0", "last_action_1"]

if len(sys.argv) != 2:
    sys.exit("usage: export_policy.py model_<n>.pt")
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = {k: v.numpy() for k, v in ckpt["actor_state_dict"].items() if hasattr(v, "numpy")}

n_obs = a["mlp.0.weight"].shape[1]
if n_obs != len(OBS_NAMES):
    sys.exit(f"this checkpoint takes {n_obs} inputs, not {len(OBS_NAMES)}: train the Drive-Straight-Lego task, "
             "not Drive-Straight")

joints = [j.get("name") for j in ET.parse(ROBOT_XML).iter("joint") if j.get("type") != "free"]
order = [j for j in joints if j in ("left_leg", "right_leg")]

np.savez(
    HERE / "policy_lego.npz",
    mean=a["obs_normalizer._mean"].reshape(-1), std=a["obs_normalizer._std"].reshape(-1) + EPS,
    w0=a["mlp.0.weight"], b0=a["mlp.0.bias"], w1=a["mlp.2.weight"], b1=a["mlp.2.bias"],
    w2=a["mlp.4.weight"], b2=a["mlp.4.bias"],
    action_order=np.array(order), obs_names=np.array(OBS_NAMES),
    max_leg_speed=LEGO_MAX_LEG_SPEED, control_hz=LEGO_CONTROL_HZ,
    iteration=ckpt.get("iter", -1),
)
print(f"wrote {HERE / 'policy_lego.npz'}  (iteration {ckpt.get('iter')}, outputs = {order})")
