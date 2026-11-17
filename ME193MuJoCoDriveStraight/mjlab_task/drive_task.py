"""mjlab task: teach the Onshape two-wheel robot to drive in a straight line down +x.

The robot starts at the origin pointing a little off to one side (random yaw) and must drive forward
fast while staying on the line y = 0 and keeping its heading along +x. The policy sees its own
velocities, its heading error and its sideways offset, and outputs one wheel speed for each wheel.

Importing this file registers the task "Drive-Straight" (see train.py / play.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  bad_orientation,
  joint_vel_rel,
  last_action,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# the Onshape export: next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "onshape" / "robot.xml", _HERE.parent / "onshape" / "robot.xml") if p.exists())
LEFT, RIGHT = "left_leg", "right_leg"       # joint names in robot.xml
TARGET_SPEED = 0.1          # m/s the robot is rewarded for going forward (sillywalk does ~0.08 m/s at 8 rad/s)
FWD, SIDE = 1, 0            # world axis the robot walks along (1 = +y: the leg axles point along x) and the sideways one
MAX_WHEEL_SPEED = 10.0      # rad/s; a policy output of 1.0 means this wheel speed
WHEEL_KD = 0.05             # how hard the motor pushes toward the wheel speed it was told (N*m per rad/s)
WHEEL_TORQUE_LIMIT = 0.3    # N*m
START_YAW = 0.4             # rad; the robot starts pointing up to this far off +x
MAX_OFF_LINE = 1.5          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0
# -------------------------------------------------------------------------------------------------------

_ROBOT = SceneEntityCfg("robot")


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  for actuator in list(spec.actuators):     # mjlab adds its own wheel motors (below)
    spec.delete(actuator)
  chassis = spec.worldbody.first_body()     # the Onshape root part (part_1); renamed so the viewer can find it
  chassis.name = "chassis"
  if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
    chassis.add_freejoint()                 # let the robot move around (same as drive.py)
  for g in chassis.geoms:                   # only the wheels should grip; the chassis slides (same as drive.py)
    g.friction[0] = 0.01
  for leg in chassis.bodies:                # name the leg geoms so friction can be randomized per leg (Lego task)
    for i, g in enumerate(leg.geoms):
      g.name = f"{leg.name}_leg_geom_{i}"
  return spec


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.03), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the velocity we ask for
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=WHEEL_KD, effort_limit=WHEEL_TORQUE_LIMIT
        ),
      ),
    ),
  )


# ---- observations and rewards: how the robot is doing relative to "straight down +x" -------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line, in metres."""
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, SIDE] - env.scene.env_origins[:, SIDE]


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw), off-line distance]. Shape [num_envs, 3]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), _off_line(env)], dim=-1)


def forward_speed(env: ManagerBasedRlEnv, target: float = TARGET_SPEED) -> torch.Tensor:
  """Reward for going along +x at TARGET_SPEED: 0 when standing still, 1 at the target, less beyond it, negative backwards.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, FWD]
  return torch.clamp(1.0 - torch.square((vx - target) / target), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_off_line(env))


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def drive_straight_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "wheel_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "wheels": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_WHEEL_SPEED, use_default_offset=True
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
  }

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=1.0),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.05),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(60.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=1.2,
      elevation=-25.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
    decimation=4,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def drive_straight_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="drive_straight",
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
  )


register_mjlab_task(
  task_id="Drive-Straight",
  env_cfg=drive_straight_env_cfg(),
  play_env_cfg=drive_straight_env_cfg(play=True),
  rl_cfg=drive_straight_ppo_runner_cfg(),
)


# ==== Drive-Straight-Lego: same robot, but the policy only sees what the real LEGO Double Motor measures ===========
# Measured on the real car with ../lego/hw_calibrate.py (2026-11-15):
#   100% LEGO speed = 15.9 rad/s (0.165 rad/s per %, linear, both legs equal)
#   Bluetooth read + command loop = 15.9 Hz        -> the policy runs at 10 Hz here and on the car
#   hub yaw in 0.1 deg, counter-clockwise positive  -> same sign as the sim, no flip
# The actor sees 5 numbers: sin(yaw), cos(yaw), yaw rate, last action (2). No position or body velocity: the car
# cannot measure them. The critic (training only) still sees everything.
LEGO_CONTROL_HZ = 10
LEGO_MAX_LEG_SPEED = 12.0   # rad/s at action 1.0 (= 73% LEGO speed; the car tops out at 15.9)
LEGO_TARGET_SPEED = 0.08    # m/s


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def yaw_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World z angular velocity, rad/s (on the car: change in hub yaw per loop)."""
  return env.scene["robot"].data.root_link_ang_vel_w[:, 2:3]


def drive_straight_lego_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = drive_straight_env_cfg(play)
  cfg.observations["actor"] = ObservationGroupCfg(
    {   # order matters: ../lego/lego_run.py builds the same 5 numbers in this order
      "heading": ObservationTermCfg(func=heading, noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03)),
      "yaw_rate": ObservationTermCfg(func=yaw_rate, noise=UniformNoiseCfg(n_min=-0.15, n_max=0.15),
                                     delay_min_lag=0, delay_max_lag=1),   # Bluetooth is up to one step late
      "last_action": ObservationTermCfg(func=last_action),
    },
    enable_corruption=not play,
  )
  cfg.actions["wheels"].scale = LEGO_MAX_LEG_SPEED
  cfg.decimation = round(1.0 / (LEGO_CONTROL_HZ * cfg.sim.mujoco.timestep))   # 20 physics steps per command
  cfg.rewards["forward_speed"].params = {"target": LEGO_TARGET_SPEED}
  cfg.rewards["face_forward"].weight = 1.0      # heading is the one thing the car can see, so lean on it
  cfg.rewards["stay_on_line"].weight = 0.5
  # domain randomization: every robot gets slightly different legs, so the policy copes with the real car's curving
  cfg.events["leg_friction"] = EventTermCfg(
    mode="startup", func=dr.geom_friction,
    params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*leg_geom.*",)), "operation": "abs", "ranges": (0.4, 1.2)},
  )
  cfg.events["motor_strength"] = EventTermCfg(
    mode="startup", func=dr.effort_limits,
    params={"asset_cfg": SceneEntityCfg("robot"), "effort_limit_range": (0.7, 1.3)},   # all motors = the two legs
  )
  return cfg


def drive_straight_lego_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = drive_straight_ppo_runner_cfg()
  cfg.experiment_name = "drive_straight_lego"
  cfg.clip_actions = 1.0      # the car gets at most LEGO_MAX_LEG_SPEED
  cfg.max_iterations = 1000
  return cfg


register_mjlab_task(
  task_id="Drive-Straight-Lego",
  env_cfg=drive_straight_lego_env_cfg(),
  play_env_cfg=drive_straight_lego_env_cfg(play=True),
  rl_cfg=drive_straight_lego_ppo_runner_cfg(),
)
