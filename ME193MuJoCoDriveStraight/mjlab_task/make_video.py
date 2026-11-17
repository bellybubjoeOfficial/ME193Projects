"""Record a video of a trained Drive-Straight policy. Made for the cluster, which has no screen.

    python3 runIt.py --gpu --time 00:10:00 --mem 16G \
      --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
      --with runs/<your run>/logs/rsl_rl/drive_straight/<date folder>/model_499.pt \
      09_onshape_robot/mjlab_task/make_video.py model_499.pt

runIt.py brings the .mp4 home to ./runs/ and opens it. (mjlab's own play.py also records a video, but then opens a
web viewer that never exits on a machine without a screen.)
"""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import drive_task  # registers "Drive-Straight" and "Drive-Straight-Lego"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

SECONDS = 10

if len(sys.argv) not in (2, 3):
  sys.exit("usage: make_video.py model_499.pt [Drive-Straight | Drive-Straight-Lego]")
checkpoint = Path(sys.argv[1])
TASK = sys.argv[2] if len(sys.argv) == 3 else "Drive-Straight"

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.viewer.width, env_cfg.viewer.height = 640, 480
agent_cfg = load_rl_cfg(TASK)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
HZ = round(1.0 / env.step_dt)   # 50 for Drive-Straight, 10 for Drive-Straight-Lego
STEPS = SECONDS * HZ
env = VideoRecorder(env, video_folder=".", step_trigger=lambda step: step == 0, video_length=STEPS, disable_logger=True)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.unwrapped.scene["robot"]
print("  second   forward (m)   sideways (m)   heading (deg)   speed (m/s)")
obs = env.get_observations()
with torch.no_grad():
  for step in range(STEPS):
    obs = env.step(policy(obs))[0]
    if step % HZ == HZ - 1:   # once a second
      fwd = robot.data.root_link_pos_w[0, drive_task.FWD].item()
      side = robot.data.root_link_pos_w[0, drive_task.SIDE].item()
      w, qx, qy, qz = robot.data.root_link_quat_w[0].tolist()
      yaw = torch.atan2(torch.tensor(2 * (w * qz + qx * qy)), torch.tensor(1 - 2 * (qy * qy + qz * qz))).item()
      v = robot.data.root_link_lin_vel_w[0, drive_task.FWD].item()
      print(f"  {(step + 1) / HZ:5.0f}  {fwd:12.2f}  {side:13.2f}  {yaw * 57.296:13.1f}  {v:12.3f}")
env.close()
print("wrote the video (an .mp4 next to this script)")
