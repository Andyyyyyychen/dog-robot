# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame.

    Uses exponential kernel for reward computation.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


def stuck_with_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize commanded robots that are nearly not moving."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_xy = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_vel_xy = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.logical_and(cmd_xy > command_threshold, body_vel_xy < velocity_threshold).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def yaw_stuck_with_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    yaw_velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize commanded yaw turns that produce almost no body yaw rate."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_yaw = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    body_yaw = torch.abs(asset.data.root_ang_vel_b[:, 2])
    reward = torch.logical_and(cmd_yaw > command_threshold, body_yaw < yaw_velocity_threshold).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def yaw_command_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    max_yaw_rate: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed yaw-rate progress when a turn command is present."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_yaw = env.command_manager.get_command(command_name)[:, 2]
    signed_yaw_rate = torch.sign(command_yaw) * asset.data.root_ang_vel_b[:, 2]
    reward = torch.clamp(signed_yaw_rate / max(max_yaw_rate, 1.0e-6), min=0.0, max=1.0)
    reward *= (torch.abs(command_yaw) > command_threshold).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def yaw_wheel_differential_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    max_xy_command: float,
    max_yaw_rate: float,
    target_wheel_diff: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward in-place yaw that is produced with left/right wheel speed difference."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_yaw = command[:, 2]
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_active = torch.logical_and(
        torch.abs(command_yaw) > command_threshold,
        command_xy_norm <= max_xy_command,
    )

    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    left_wheel_vel = torch.mean(wheel_vel[:, [0, 2]], dim=1)
    right_wheel_vel = torch.mean(wheel_vel[:, [1, 3]], dim=1)
    wheel_diff_score = torch.clamp(
        torch.abs(right_wheel_vel - left_wheel_vel) / max(target_wheel_diff, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    signed_yaw_rate = torch.sign(command_yaw) * asset.data.root_ang_vel_b[:, 2]
    yaw_progress_score = torch.clamp(signed_yaw_rate / max(max_yaw_rate, 1.0e-6), min=0.0, max=1.0)
    reward = wheel_diff_score * yaw_progress_score * yaw_command_active.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def yaw_wheel_velocity_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    max_xy_command: float,
    target_wheel_diff: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward left/right wheel velocity difference that matches the commanded yaw direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_yaw = command[:, 2]
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_active = torch.logical_and(
        torch.abs(command_yaw) > command_threshold,
        command_xy_norm <= max_xy_command,
    )

    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    left_wheel_vel = torch.mean(wheel_vel[:, [0, 2]], dim=1)
    right_wheel_vel = torch.mean(wheel_vel[:, [1, 3]], dim=1)
    signed_wheel_diff = torch.sign(command_yaw) * (right_wheel_vel - left_wheel_vel)
    reward = torch.clamp(signed_wheel_diff / max(target_wheel_diff, 1.0e-6), min=0.0, max=1.0)
    reward *= yaw_command_active.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def commanded_base_height_below_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    height_margin: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    straight_command_only: bool = False,
    max_abs_yaw_command: float | None = None,
) -> torch.Tensor:
    """Penalize crouching below the target base height while a velocity command is active."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_abs = torch.abs(command[:, 2])
    active_command = torch.logical_or(
        command_xy_norm > command_threshold,
        yaw_command_abs > command_threshold,
    )
    if straight_command_only:
        yaw_limit = command_threshold if max_abs_yaw_command is None else max_abs_yaw_command
        active_command = torch.logical_and(command_xy_norm > command_threshold, yaw_command_abs <= yaw_limit)
    height_deficit = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    reward = torch.square(height_deficit / max(height_margin, 1.0e-6))
    reward *= active_command.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def commanded_joint_posture_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    straight_command_only: bool = False,
    max_abs_yaw_command: float | None = None,
    yaw_command_only: bool = False,
    max_xy_command: float | None = None,
) -> torch.Tensor:
    """Penalize commanded motion that bends selected joints away from the default posture."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_abs = torch.abs(command[:, 2])
    active_command = torch.logical_or(
        command_xy_norm > command_threshold,
        yaw_command_abs > command_threshold,
    )
    if straight_command_only:
        yaw_limit = command_threshold if max_abs_yaw_command is None else max_abs_yaw_command
        active_command = torch.logical_and(command_xy_norm > command_threshold, yaw_command_abs <= yaw_limit)
    if yaw_command_only:
        xy_limit = command_threshold if max_xy_command is None else max_xy_command
        active_command = torch.logical_and(yaw_command_abs > command_threshold, command_xy_norm <= xy_limit)

    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.mean(torch.square(joint_error), dim=1)
    reward *= active_command.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def yaw_hipx_twist_without_yaw_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    max_xy_command: float,
    yaw_velocity_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize hipx twisting during yaw commands when the body is not actually turning."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_yaw = command[:, 2]
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_active = torch.logical_and(
        torch.abs(command_yaw) > command_threshold,
        command_xy_norm <= max_xy_command,
    )

    signed_yaw_rate = torch.sign(command_yaw) * asset.data.root_ang_vel_b[:, 2]
    poor_yaw_progress = torch.clamp(
        (yaw_velocity_threshold - signed_yaw_rate) / max(yaw_velocity_threshold, 1.0e-6),
        min=0.0,
        max=1.0,
    )
    hipx_error = torch.mean(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )
    reward = hipx_error * poor_yaw_progress * yaw_command_active.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_spin_when_stuck(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize wheel spinning when a commanded robot is barely translating."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd_xy = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_vel_xy = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    wheel_speed = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    is_stuck = torch.logical_and(cmd_xy > command_threshold, body_vel_xy < velocity_threshold)
    reward = wheel_speed * is_stuck.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def adaptive_lateral_force_ratio(
    forces_z: torch.Tensor,
    base_lateral_force_ratio: float,
    min_lateral_force_ratio: float,
    max_lateral_force_ratio: float,
    vertical_force_eps: float,
) -> torch.Tensor:
    """Scale the lateral/vertical contact-force ratio with the current support load."""
    support_load = torch.mean(torch.clamp(forces_z, min=vertical_force_eps), dim=1)
    reference_load = torch.median(support_load.detach()).clamp(min=vertical_force_eps)
    ratio = base_lateral_force_ratio * torch.sqrt(reference_load / torch.clamp(support_load, min=vertical_force_eps))
    return torch.clamp(ratio, min=min_lateral_force_ratio, max=max_lateral_force_ratio)


def wheel_spin_with_lateral_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    lateral_force_ratio: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_lateral_force_ratio: float | None = None,
    min_lateral_force_ratio: float = 1.3,
    max_lateral_force_ratio: float = 3.0,
    vertical_force_eps: float = 5.0,
) -> torch.Tensor:
    """Penalize wheel spinning against vertical edges such as stair risers."""
    if base_lateral_force_ratio is None:
        base_lateral_force_ratio = lateral_force_ratio if lateral_force_ratio is not None else 2.0

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    adaptive_ratio = adaptive_lateral_force_ratio(
        forces_z, base_lateral_force_ratio, min_lateral_force_ratio, max_lateral_force_ratio, vertical_force_eps
    )
    vertical_edge_contact = torch.any(
        forces_xy > adaptive_ratio.unsqueeze(1) * torch.clamp(forces_z, min=vertical_force_eps), dim=1
    )
    wheel_speed = torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    reward = wheel_speed * vertical_edge_contact.float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def _upright_scale(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a 0..1 scale that fades rewards out when the base tips over."""
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def _bounded_phase_score(time: torch.Tensor, min_time: float, max_time: float | None = None) -> torch.Tensor:
    """Score a contact/air phase while discouraging phases that last too long."""
    score = torch.clamp(time / max(min_time, 1.0e-6), min=0.0, max=1.0)
    if max_time is not None:
        score *= torch.clamp((max_time - time) / max(max_time - min_time, 1.0e-6), min=0.0, max=1.0)
    return score


def _root_step_delta(
    env: ManagerBasedRLEnv, asset: RigidObject, cache_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute root displacement since the previous reward call in world and yaw frames."""
    current_root_pos = asset.data.root_pos_w[:, :3]
    if not hasattr(env, cache_name) or getattr(env, cache_name).shape != current_root_pos.shape:
        setattr(env, cache_name, current_root_pos.detach().clone())
        zeros = torch.zeros_like(current_root_pos)
        return zeros, zeros

    previous_root_pos = getattr(env, cache_name)
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_envs = env.episode_length_buf <= 1
        previous_root_pos = torch.where(reset_envs.unsqueeze(1), current_root_pos, previous_root_pos)
    else:
        reset_envs = None

    delta_w = current_root_pos - previous_root_pos
    delta_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    if reset_envs is not None:
        delta_w = torch.where(reset_envs.unsqueeze(1), torch.zeros_like(delta_w), delta_w)
        delta_b = torch.where(reset_envs.unsqueeze(1), torch.zeros_like(delta_b), delta_b)

    setattr(env, cache_name, current_root_pos.detach().clone())
    return delta_w, delta_b


def _root_window_delta(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    cache_name: str,
    window_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute root displacement over a short rolling window."""
    window_steps = max(int(window_steps), 1)
    current_root_pos = asset.data.root_pos_w[:, :3]
    pos_cache_name = f"{cache_name}_root_pos_w"
    counter_cache_name = f"{cache_name}_counter"

    if (
        not hasattr(env, pos_cache_name)
        or getattr(env, pos_cache_name).shape != current_root_pos.shape
        or not hasattr(env, counter_cache_name)
        or getattr(env, counter_cache_name).shape[0] != env.num_envs
    ):
        setattr(env, pos_cache_name, current_root_pos.detach().clone())
        setattr(env, counter_cache_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
        zeros = torch.zeros_like(current_root_pos)
        return zeros, zeros

    previous_root_pos = getattr(env, pos_cache_name)
    counter = getattr(env, counter_cache_name)
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_envs = env.episode_length_buf <= 1
        previous_root_pos = torch.where(reset_envs.unsqueeze(1), current_root_pos, previous_root_pos)
        counter = torch.where(reset_envs, torch.zeros_like(counter), counter)
    else:
        reset_envs = None

    delta_w = current_root_pos - previous_root_pos
    delta_b = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), delta_w)
    if reset_envs is not None:
        delta_w = torch.where(reset_envs.unsqueeze(1), torch.zeros_like(delta_w), delta_w)
        delta_b = torch.where(reset_envs.unsqueeze(1), torch.zeros_like(delta_b), delta_b)

    update_envs = counter >= window_steps
    next_root_pos = torch.where(update_envs.unsqueeze(1), current_root_pos, previous_root_pos)
    next_counter = torch.where(update_envs, torch.zeros_like(counter), counter + 1)
    setattr(env, pos_cache_name, next_root_pos.detach().clone())
    setattr(env, counter_cache_name, next_counter.detach().clone())
    return delta_w, delta_b


def commanded_motion_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    max_progress: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward real displacement in the commanded xy direction.

    Velocity tracking alone can look good while the robot is pinned against a
    stair edge. This term asks for actual per-step translation in the requested
    direction, including reverse commands.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    _, delta_b = _root_step_delta(env, asset, "_commanded_motion_progress_prev_root_pos_w")
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.norm(command_xy, dim=1)
    projected_progress = torch.sum(delta_b[:, :2] * command_xy, dim=1) / torch.clamp(command_norm, min=1.0e-6)
    reward = torch.clamp(projected_progress / max_progress, min=0.0, max=1.0)
    reward *= (command_norm > command_threshold).float()
    reward *= _upright_scale(env)
    return reward


def stair_upward_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    max_forward_step: float,
    max_up_step: float,
    min_forward_step: float = 0.0,
    min_up_step: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_steps: int = 10,
    forward_weight: float = 0.0,
    coupled_weight: float = 0.85,
    upward_weight: float = 0.15,
    min_forward_fraction: float = 0.15,
) -> torch.Tensor:
    """Reward stable forward progress that also gains height on stair-like terrain."""
    asset: RigidObject = env.scene[asset_cfg.name]
    delta_w, delta_b = _root_window_delta(env, asset, "_stair_upward_progress", window_steps)
    forward_command = env.command_manager.get_command(command_name)[:, 0]
    forward_span = max(max_forward_step - min_forward_step, 1.0e-6)
    upward_span = max(max_up_step - min_up_step, 1.0e-6)
    forward_progress = torch.clamp((delta_b[:, 0] - min_forward_step) / forward_span, min=0.0, max=1.0)
    upward_progress = torch.clamp((delta_w[:, 2] - min_up_step) / upward_span, min=0.0, max=1.0)

    coupled_progress = torch.clamp(forward_progress * upward_progress, min=0.0)
    smooth_coupled_progress = torch.sqrt(coupled_progress)
    forward_gate = torch.clamp(forward_progress / max(min_forward_fraction, 1.0e-6), min=0.0, max=1.0)
    gated_upward_progress = upward_progress * forward_gate
    weight_sum = max(forward_weight + coupled_weight + upward_weight, 1.0e-6)
    reward = (
        forward_weight * forward_progress
        + coupled_weight * smooth_coupled_progress
        + upward_weight * gated_upward_progress
    ) / weight_sum
    reward *= (forward_command > command_threshold).float()
    reward *= _upright_scale(env)
    return reward


def upward_without_forward_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    max_up_step: float,
    min_forward_step: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_steps: int = 30,
) -> torch.Tensor:
    """Penalize gaining height without enough net forward progress.

    This prevents the policy from scoring stair rewards by bouncing the body,
    pitching into an edge, or being squeezed upward while not actually climbing.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    delta_w, delta_b = _root_window_delta(env, asset, "_upward_without_forward_progress", window_steps)
    forward_command = env.command_manager.get_command(command_name)[:, 0]
    upward_progress = torch.clamp(delta_w[:, 2] / max(max_up_step, 1.0e-6), min=0.0, max=1.0)
    missing_forward = torch.clamp((min_forward_step - delta_b[:, 0]) / max(min_forward_step, 1.0e-6), min=0.0, max=1.0)
    reward = upward_progress * missing_forward
    reward *= (forward_command > command_threshold).float()
    reward *= _upright_scale(env)
    return reward


def vertical_bounce_without_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    velocity_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical bouncing when forward commands are not producing progress."""
    asset: RigidObject = env.scene[asset_cfg.name]
    forward_command = env.command_manager.get_command(command_name)[:, 0]
    forward_speed = asset.data.root_lin_vel_b[:, 0]
    no_progress_scale = torch.clamp((velocity_threshold - forward_speed) / velocity_threshold, min=0.0, max=1.0)
    reward = torch.abs(asset.data.root_lin_vel_b[:, 2]) * no_progress_scale
    reward *= (forward_command > command_threshold).float()
    reward *= _upright_scale(env)
    return reward


def wheel_spin_without_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    velocity_threshold: float,
    wheel_speed_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize fast wheel spinning while commanded translation is not happening."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_xy = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_vel_xy = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    wheel_speed = torch.mean(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stuck_scale = torch.clamp((velocity_threshold - body_vel_xy) / velocity_threshold, min=0.0, max=1.0)
    spin_excess = torch.clamp(wheel_speed - wheel_speed_threshold, min=0.0)
    reward = spin_excess * stuck_scale
    reward *= (command_xy > command_threshold).float()
    reward *= _upright_scale(env)
    return reward


def wheel_lateral_edge_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    base_lateral_force_ratio: float,
    min_lateral_force_ratio: float,
    max_lateral_force_ratio: float,
    vertical_force_eps: float,
) -> torch.Tensor:
    """Penalize wheel contacts that mostly push sideways into vertical edges."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    adaptive_ratio = adaptive_lateral_force_ratio(
        forces_z, base_lateral_force_ratio, min_lateral_force_ratio, max_lateral_force_ratio, vertical_force_eps
    )
    lateral_ratio = forces_xy / torch.clamp(forces_z, min=vertical_force_eps)
    edge_excess = torch.clamp(lateral_ratio - adaptive_ratio.unsqueeze(1), min=0.0, max=3.0)
    reward = torch.sum(edge_excess, dim=1)
    reward *= _upright_scale(env)
    return reward


def wheel_clearance_on_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    min_height: float,
    target_height: float,
    tanh_mult: float,
    forward_velocity_threshold: float = 0.08,
    min_progress_scale: float = 0.35,
) -> torch.Tensor:
    """Reward commanded motion with wheels lifted in the body frame.

    The reward is bounded to 0..1 and only active while commands are present.
    It gives the policy a direct stair-climbing hint: lift wheels above the
    normal rolling height while they are moving, instead of scraping stair edges.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    foot_vel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)

    foot_pos_b = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    foot_vel_b = torch.zeros_like(foot_pos_b)
    for i in range(len(asset_cfg.body_ids)):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_pos_translated[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_vel_translated[:, i, :])

    height_span = max(target_height - min_height, 1.0e-6)
    clearance = torch.clamp((foot_pos_b[:, :, 2] - min_height) / height_span, min=0.0, max=1.0)
    swing_scale = torch.tanh(tanh_mult * torch.linalg.norm(foot_vel_b[:, :, :2], dim=2))
    forward_speed_scale = torch.clamp(asset.data.root_lin_vel_b[:, 0] / forward_velocity_threshold, min=0.0, max=1.0)
    progress_scale = min_progress_scale + (1.0 - min_progress_scale) * forward_speed_scale
    reward = torch.mean(clearance * swing_scale, dim=1) * progress_scale

    command = env.command_manager.get_command(command_name)
    reward *= (command[:, 0] > command_threshold).float()
    reward *= _upright_scale(env)
    return reward


def yaw_turn_feet_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    max_xy_command: float,
    min_height: float,
    target_height: float,
    min_air_time: float,
    max_air_time: float,
    tanh_mult: float,
    min_base_height: float,
    base_height_margin: float,
    synced_feet_pair_names: tuple[tuple[str, str], tuple[str, str]] | None = None,
    min_contact_time: float = 0.01,
    diagonal_pair_weight: float = 0.85,
) -> torch.Tensor:
    """Reward diagonal wheel clearance during near-in-place yaw turns."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_abs = torch.abs(command[:, 2])
    active_command = torch.logical_and(yaw_command_abs > command_threshold, command_xy_norm <= max_xy_command)

    use_diagonal_pairs = synced_feet_pair_names is not None
    if use_diagonal_pairs:
        asset_pair_0 = list(asset.find_bodies(synced_feet_pair_names[0])[0])
        asset_pair_1 = list(asset.find_bodies(synced_feet_pair_names[1])[0])
        sensor_pair_0 = list(contact_sensor.find_bodies(synced_feet_pair_names[0])[0])
        sensor_pair_1 = list(contact_sensor.find_bodies(synced_feet_pair_names[1])[0])
        asset_body_ids = asset_pair_0 + asset_pair_1
        sensor_body_ids = sensor_pair_0 + sensor_pair_1
    else:
        asset_body_ids = asset_cfg.body_ids
        sensor_body_ids = sensor_cfg.body_ids

    foot_pos_translated = asset.data.body_pos_w[:, asset_body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    foot_vel_translated = asset.data.body_lin_vel_w[:, asset_body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)

    foot_pos_b = torch.zeros(env.num_envs, len(asset_body_ids), 3, device=env.device)
    foot_vel_b = torch.zeros_like(foot_pos_b)
    for i in range(len(asset_body_ids)):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_pos_translated[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_vel_translated[:, i, :])

    height_span = max(target_height - min_height, 1.0e-6)
    clearance = torch.clamp((foot_pos_b[:, :, 2] - min_height) / height_span, min=0.0, max=1.0)
    swing_scale = torch.tanh(tanh_mult * torch.linalg.norm(foot_vel_b[:, :, :2], dim=2))
    air_time = contact_sensor.data.current_air_time[:, sensor_body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_body_ids]
    air_score = torch.clamp(air_time / max(min_air_time, 1.0e-6), min=0.0, max=1.0)
    contact_score = torch.clamp(contact_time / max(min_contact_time, 1.0e-6), min=0.0, max=1.0)
    short_air = torch.clamp((max_air_time - air_time) / max(max_air_time - min_air_time, 1.0e-6), min=0.0, max=1.0)
    base_height_scale = torch.clamp(
        (asset.data.root_pos_w[:, 2] - min_base_height) / max(base_height_margin, 1.0e-6), min=0.0, max=1.0
    )

    lift_score = clearance * swing_scale * air_score * short_air
    reward = torch.mean(lift_score, dim=1)
    if use_diagonal_pairs:
        pair_0_lift = torch.mean(lift_score[:, 0:2], dim=1)
        pair_1_lift = torch.mean(lift_score[:, 2:4], dim=1)
        pair_0_ground = torch.mean(contact_score[:, 0:2], dim=1)
        pair_1_ground = torch.mean(contact_score[:, 2:4], dim=1)
        pair_0_swing_phase = pair_0_lift * pair_1_ground
        pair_1_swing_phase = pair_1_lift * pair_0_ground
        diagonal_reward = torch.maximum(pair_0_swing_phase, pair_1_swing_phase)
        reward = diagonal_pair_weight * diagonal_reward + (1.0 - diagonal_pair_weight) * reward
    reward *= active_command.float()
    reward *= base_height_scale
    reward *= _upright_scale(env)
    return reward


def yaw_turn_diagonal_step(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    max_xy_command: float,
    synced_feet_pair_names: tuple[tuple[str, str], tuple[str, str]],
    min_air_time: float,
    min_contact_time: float,
    phase_balance_weight: float,
    max_air_time: float | None = None,
    max_contact_time: float | None = None,
) -> torch.Tensor:
    """Reward diagonal swing/support phases during near-in-place yaw turns."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_abs = torch.abs(command[:, 2])
    active_command = torch.logical_and(yaw_command_abs > command_threshold, command_xy_norm <= max_xy_command)

    pair_0 = list(contact_sensor.find_bodies(synced_feet_pair_names[0])[0])
    pair_1 = list(contact_sensor.find_bodies(synced_feet_pair_names[1])[0])
    body_ids = pair_0 + pair_1
    air_time = contact_sensor.data.current_air_time[:, body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, body_ids]

    pair_0_air = torch.mean(_bounded_phase_score(air_time[:, 0:2], min_air_time, max_air_time), dim=1)
    pair_1_air = torch.mean(_bounded_phase_score(air_time[:, 2:4], min_air_time, max_air_time), dim=1)
    pair_0_contact = torch.mean(
        _bounded_phase_score(contact_time[:, 0:2], min_contact_time, max_contact_time), dim=1
    )
    pair_1_contact = torch.mean(
        _bounded_phase_score(contact_time[:, 2:4], min_contact_time, max_contact_time), dim=1
    )

    pair_0_swing = pair_0_air * pair_1_contact
    pair_1_swing = pair_1_air * pair_0_contact
    diagonal_phase = torch.maximum(pair_0_swing, pair_1_swing)
    phase_balance = torch.clamp(torch.abs(pair_0_air - pair_1_air), min=0.0, max=1.0)
    reward = diagonal_phase * (phase_balance_weight + (1.0 - phase_balance_weight) * phase_balance)
    reward *= active_command.float()
    reward *= _upright_scale(env)
    return reward


def yaw_turn_air_time_deficit(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    max_xy_command: float,
    synced_feet_pair_names: tuple[tuple[str, str], tuple[str, str]],
    min_air_time: float,
    min_contact_time: float,
    max_air_time: float | None = None,
    max_contact_time: float | None = None,
) -> torch.Tensor:
    """Penalize yaw turns that do not create a diagonal swing/support phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_abs = torch.abs(command[:, 2])
    active_command = torch.logical_and(yaw_command_abs > command_threshold, command_xy_norm <= max_xy_command)

    pair_0 = list(contact_sensor.find_bodies(synced_feet_pair_names[0])[0])
    pair_1 = list(contact_sensor.find_bodies(synced_feet_pair_names[1])[0])
    body_ids = pair_0 + pair_1
    air_time = contact_sensor.data.current_air_time[:, body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, body_ids]

    pair_0_air = torch.mean(_bounded_phase_score(air_time[:, 0:2], min_air_time, max_air_time), dim=1)
    pair_1_air = torch.mean(_bounded_phase_score(air_time[:, 2:4], min_air_time, max_air_time), dim=1)
    pair_0_contact = torch.mean(
        _bounded_phase_score(contact_time[:, 0:2], min_contact_time, max_contact_time), dim=1
    )
    pair_1_contact = torch.mean(
        _bounded_phase_score(contact_time[:, 2:4], min_contact_time, max_contact_time), dim=1
    )

    pair_0_swing_support = pair_0_air * pair_1_contact
    pair_1_swing_support = pair_1_air * pair_0_contact
    best_diagonal_phase = torch.maximum(pair_0_swing_support, pair_1_swing_support)
    penalty = torch.square(1.0 - best_diagonal_phase)
    penalty *= active_command.float()
    penalty *= _upright_scale(env)
    return penalty


def yaw_turn_phase_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    max_xy_command: float,
    synced_feet_pair_names: tuple[tuple[str, str], tuple[str, str]],
    max_air_time: float,
    max_contact_time: float,
) -> torch.Tensor:
    """Penalize yaw turns that hold the same diagonal swing/support phase too long."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command_abs = torch.abs(command[:, 2])
    active_command = torch.logical_and(yaw_command_abs > command_threshold, command_xy_norm <= max_xy_command)

    pair_0 = list(contact_sensor.find_bodies(synced_feet_pair_names[0])[0])
    pair_1 = list(contact_sensor.find_bodies(synced_feet_pair_names[1])[0])
    body_ids = pair_0 + pair_1
    air_time = contact_sensor.data.current_air_time[:, body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, body_ids]

    air_timeout = torch.relu(air_time - max_air_time) / max(max_air_time, 1.0e-6)
    contact_timeout = torch.relu(contact_time - max_contact_time) / max(max_contact_time, 1.0e-6)
    penalty = torch.mean(torch.square(air_timeout) + torch.square(contact_timeout), dim=1)
    penalty *= active_command.float()
    penalty *= _upright_scale(env)
    return penalty


def yaw_turn_tangential_swing(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float,
    max_xy_command: float,
    min_height: float,
    target_height: float,
    min_air_time: float,
    max_air_time: float,
    min_contact_time: float,
    max_contact_time: float,
    tanh_mult: float,
    synced_feet_pair_names: tuple[tuple[str, str], tuple[str, str]],
) -> torch.Tensor:
    """Reward lifted diagonal wheels that swing in the commanded yaw direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
    yaw_command = command[:, 2]
    yaw_command_abs = torch.abs(yaw_command)
    active_command = torch.logical_and(yaw_command_abs > command_threshold, command_xy_norm <= max_xy_command)

    asset_pair_0 = list(asset.find_bodies(synced_feet_pair_names[0])[0])
    asset_pair_1 = list(asset.find_bodies(synced_feet_pair_names[1])[0])
    sensor_pair_0 = list(contact_sensor.find_bodies(synced_feet_pair_names[0])[0])
    sensor_pair_1 = list(contact_sensor.find_bodies(synced_feet_pair_names[1])[0])
    asset_body_ids = asset_pair_0 + asset_pair_1
    sensor_body_ids = sensor_pair_0 + sensor_pair_1

    foot_pos_translated = asset.data.body_pos_w[:, asset_body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    foot_vel_translated = asset.data.body_lin_vel_w[:, asset_body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    foot_pos_b = torch.zeros(env.num_envs, len(asset_body_ids), 3, device=env.device)
    foot_vel_b = torch.zeros_like(foot_pos_b)
    for i in range(len(asset_body_ids)):
        foot_pos_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_pos_translated[:, i, :])
        foot_vel_b[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, foot_vel_translated[:, i, :])

    height_span = max(target_height - min_height, 1.0e-6)
    clearance = torch.clamp((foot_pos_b[:, :, 2] - min_height) / height_span, min=0.0, max=1.0)
    air_time = contact_sensor.data.current_air_time[:, sensor_body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_body_ids]
    air_score = _bounded_phase_score(air_time, min_air_time, max_air_time)
    contact_score = _bounded_phase_score(contact_time, min_contact_time, max_contact_time)

    tangent = torch.stack((-foot_pos_b[:, :, 1], foot_pos_b[:, :, 0]), dim=2)
    tangent_norm = torch.linalg.norm(tangent, dim=2, keepdim=True)
    tangent_dir = tangent / torch.clamp(tangent_norm, min=1.0e-6)
    signed_tangent_speed = torch.sum(foot_vel_b[:, :, :2] * tangent_dir, dim=2) * torch.sign(yaw_command).unsqueeze(1)
    swing_direction_score = torch.tanh(tanh_mult * torch.clamp(signed_tangent_speed, min=0.0))

    swing_score = clearance * air_score * swing_direction_score
    pair_0_swing = torch.mean(swing_score[:, 0:2], dim=1) * torch.mean(contact_score[:, 2:4], dim=1)
    pair_1_swing = torch.mean(swing_score[:, 2:4], dim=1) * torch.mean(contact_score[:, 0:2], dim=1)
    reward = torch.maximum(pair_0_swing, pair_1_swing)
    reward *= active_command.float()
    reward *= _upright_scale(env)
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    defined in :attr:`synced_feet_pair_names` to bias the policy towards a desired gait,
    i.e trotting, bounding, or pacing. Note that this reward is only for quadrupedal gaits
    with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        yaw_command_only: bool = False,
        max_xy_command: float | None = None,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait when a matching command is present
        command = env.command_manager.get_command(command_name)
        if yaw_command_only:
            command_xy_norm = torch.linalg.norm(command[:, :2], dim=1)
            yaw_command_abs = torch.abs(command[:, 2])
            xy_limit = command_threshold if max_xy_command is None else max_xy_command
            active = torch.logical_and(yaw_command_abs > command_threshold, command_xy_norm <= xy_limit)
        else:
            cmd = torch.linalg.norm(command, dim=1)
            body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
            active = torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold)
        reward = torch.where(active, sync_reward * async_reward, 0.0)
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str | None = None,
    yaw_command_threshold: float = 0.08,
    max_xy_command: float = 0.18,
    yaw_slide_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        yaw_turn_active = torch.logical_and(
            torch.abs(command[:, 2]) > yaw_command_threshold,
            torch.linalg.norm(command[:, :2], dim=1) <= max_xy_command,
        )
        reward = torch.where(yaw_turn_active, yaw_slide_scale * reward, reward)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(
#         env.action_manager.action - 2 * env.action_manager.prev_action
#         + env.action_manager.prev_prev_action
#     )
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
