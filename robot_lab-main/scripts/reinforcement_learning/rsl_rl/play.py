# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper as IsaacLabRslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version):
        return agent_cfg

try:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
except ImportError:
    RslRlBaseRunnerCfg = Any
try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    def get_published_pretrained_checkpoint(*_args, **_kwargs):
        return None

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def rsl_rl_obs_to_tensordict(obs, extras: dict | None, num_envs: int, device) -> TensorDict:
    if isinstance(obs, TensorDict):
        return obs
    if hasattr(obs, "keys") and not isinstance(obs, torch.Tensor):
        return TensorDict(dict(obs), batch_size=[num_envs], device=device)

    obs_dict = {}
    if extras is not None and isinstance(extras.get("observations"), dict):
        obs_dict.update(extras["observations"])
    if "policy" not in obs_dict:
        obs_dict["policy"] = obs
    return TensorDict(obs_dict, batch_size=[num_envs], device=device)


class CompatRslRlVecEnvWrapper(IsaacLabRslRlVecEnvWrapper):
    """Adapt older Isaac Lab wrappers to the TensorDict API used by RSL-RL 3.x."""

    def get_observations(self):
        observations = super().get_observations()
        if isinstance(observations, tuple):
            obs, extras = observations
        else:
            obs, extras = observations, None
        return rsl_rl_obs_to_tensordict(obs, extras, self.num_envs, self.device)

    def reset(self):
        observations = super().reset()
        if isinstance(observations, tuple):
            obs, extras = observations
        else:
            obs, extras = observations, None
        return rsl_rl_obs_to_tensordict(obs, extras, self.num_envs, self.device)

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        obs = rsl_rl_obs_to_tensordict(obs, extras, self.num_envs, self.device)
        return obs, rewards, dones, extras


def get_runner_class_name(agent_cfg) -> str:
    return getattr(agent_cfg, "class_name", "OnPolicyRunner")


def get_agent_cfg_dict(agent_cfg) -> dict:
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict.setdefault("obs_groups", {"policy": ["policy"], "critic": ["critic"]})
    return agent_cfg_dict


def map_lateral_arrows_to_yaw(controller: Se2Keyboard, yaw_sensitivity: float):
    """Use arrow-left/right as yaw commands when the task does not train lateral velocity."""
    key_mapping = getattr(controller, "_INPUT_KEY_MAPPING", None)
    if key_mapping is None:
        return
    key_mapping["LEFT"] = np.asarray([0.0, 0.0, yaw_sensitivity])
    key_mapping["RIGHT"] = np.asarray([0.0, 0.0, -yaw_sensitivity])


def add_keyboard_fallback_keys(controller: Se2Keyboard, x_sensitivity: float, yaw_sensitivity: float):
    """Add WASD/QE fallbacks for remote desktops that do not forward arrow keys."""
    key_mapping = getattr(controller, "_INPUT_KEY_MAPPING", None)
    if key_mapping is None:
        return
    key_mapping["W"] = np.asarray([x_sensitivity, 0.0, 0.0])
    key_mapping["S"] = np.asarray([-x_sensitivity, 0.0, 0.0])
    key_mapping["A"] = np.asarray([0.0, 0.0, yaw_sensitivity])
    key_mapping["D"] = np.asarray([0.0, 0.0, -yaw_sensitivity])
    key_mapping["Q"] = np.asarray([0.0, 0.0, yaw_sensitivity])
    key_mapping["E"] = np.asarray([0.0, 0.0, -yaw_sensitivity])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.scene.terrain.max_init_terrain_level = 0
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        env_cfg.curriculum.terrain_levels = None
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        keyboard_yaw_sensitivity = 0.45 * env_cfg.commands.base_velocity.ranges.ang_vel_z[1]
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=keyboard_yaw_sensitivity,
        )
        controller = Se2Keyboard(config)
        add_keyboard_fallback_keys(controller, env_cfg.commands.base_velocity.ranges.lin_vel_x[1], keyboard_yaw_sensitivity)
        lin_vel_y_range = env_cfg.commands.base_velocity.ranges.lin_vel_y
        if abs(lin_vel_y_range[0]) < 1e-6 and abs(lin_vel_y_range[1]) < 1e-6:
            map_lateral_arrows_to_yaw(controller, keyboard_yaw_sensitivity)
        print(
            "[KEYBOARD] Debug enabled. Click the Isaac Sim viewport, then press arrow keys. "
            "Expected nonzero vx/yaw values will print here.",
            flush=True,
        )
        keyboard_debug_state = {"last_time": 0.0, "last_command": np.zeros(3, dtype=np.float32)}

        def keyboard_obs_term(env):
            command_np = np.asarray(controller.advance(), dtype=np.float32)
            now = time.time()
            command_changed = np.linalg.norm(command_np - keyboard_debug_state["last_command"]) > 1.0e-4
            if command_changed or now - keyboard_debug_state["last_time"] >= 1.0:
                print(
                    "[KEYBOARD] command "
                    f"vx={command_np[0]: .3f}, vy={command_np[1]: .3f}, yaw={command_np[2]: .3f}",
                    flush=True,
                )
                keyboard_debug_state["last_time"] = now
                keyboard_debug_state["last_command"] = command_np.copy()
            return torch.tensor(command_np, dtype=torch.float32).unsqueeze(0).to(env.device)

        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=keyboard_obs_term,
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = CompatRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner_class_name = get_runner_class_name(agent_cfg)
    agent_cfg_dict = get_agent_cfg_dict(agent_cfg)
    if runner_class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    elif runner_class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {runner_class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
