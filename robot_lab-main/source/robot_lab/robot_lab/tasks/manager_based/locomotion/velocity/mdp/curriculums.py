# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _episode_term_average(env: ManagerBasedRLEnv, term_name: str, env_ids: Sequence[int]) -> torch.Tensor | None:
    """Return the per-second episode average for a reward term when it exists."""
    if not term_name or term_name not in env.reward_manager._episode_sums:
        return None
    return env.reward_manager._episode_sums[term_name][env_ids] / env.max_episode_length_s


def terrain_levels_jk03_stairs(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    up_command_scale: float = 0.55,
    down_command_scale: float = 0.18,
    min_up_distance: float = 1.2,
    max_up_distance: float = 2.4,
    min_down_distance: float = 0.25,
    max_down_distance: float = 0.8,
    reward_term_name: str = "stair_upward_progress",
    min_upward_reward: float = 0.035,
    progress_term_name: str = "commanded_motion_progress",
    min_progress_reward: float = 0.08,
    max_move_up_penalty_terms: dict[str, float] | None = None,
    move_down_penalty_terms: dict[str, float] | None = None,
    min_down_upward_reward: float | None = None,
    forward_command_threshold: float = 0.08,
) -> torch.Tensor:
    """Terrain curriculum tuned for slow stair climbing.

    JK03 can move across rough terrain without actually climbing a stair.  Gate
    upward curriculum progress with stair progress and contact-quality rewards
    so terrain levels do not rise only because the robot covered xy distance.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    base_velocity_command = env.command_manager.get_command("base_velocity")[env_ids]
    command_xy = torch.linalg.norm(base_velocity_command[:, :2], dim=1)
    forward_commanded = base_velocity_command[:, 0] > forward_command_threshold

    distance = torch.linalg.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    expected_distance = command_xy * env.max_episode_length_s
    move_up_distance = torch.clamp(expected_distance * up_command_scale, min=min_up_distance, max=max_up_distance)
    move_down_distance = torch.clamp(
        expected_distance * down_command_scale, min=min_down_distance, max=max_down_distance
    )

    move_up = distance > move_up_distance
    upward_reward = _episode_term_average(env, reward_term_name, env_ids)
    if upward_reward is not None:
        move_up *= upward_reward > min_upward_reward
    progress_reward = _episode_term_average(env, progress_term_name, env_ids)
    if progress_reward is not None:
        move_up *= progress_reward > min_progress_reward
    if max_move_up_penalty_terms is not None:
        for term_name, max_penalty in max_move_up_penalty_terms.items():
            penalty = _episode_term_average(env, term_name, env_ids)
            if penalty is not None:
                move_up *= penalty > -abs(max_penalty)

    move_down = distance < move_down_distance
    bad_forward_stair_behavior = torch.zeros_like(move_down)
    if min_down_upward_reward is not None and upward_reward is not None:
        bad_forward_stair_behavior |= upward_reward < min_down_upward_reward
    if move_down_penalty_terms is not None:
        for term_name, max_penalty in move_down_penalty_terms.items():
            penalty = _episode_term_average(env, term_name, env_ids)
            if penalty is not None:
                bad_forward_stair_behavior |= penalty < -abs(max_penalty)
    move_down |= bad_forward_stair_behavior & forward_commanded
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
