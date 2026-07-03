# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import JK04RoughEnvCfg


@configclass
class JK04FlatEnvCfg(JK04RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # Flat pretraining should first learn clean forward, lateral, and yaw commands.
        # Keep the JK04 initial pose unchanged, but remove rough-terrain early
        # penalties that can make the first flat policy prefer not moving.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
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

        self.actions.joint_pos.scale = {
            ".*_hipx_joint": 0.06,
            ".*_hipy_joint": 0.24,
            ".*_knee_joint": 0.24,
        }
        self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}
        self.actions.joint_vel.scale = 8.0

        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.15
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.56
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.yaw_command_progress.weight = 0.2
        self.rewards.yaw_command_progress.params["command_threshold"] = 0.05
        self.rewards.yaw_command_progress.params["max_yaw_rate"] = 0.8
        self.rewards.yaw_wheel_differential_progress.weight = 0
        self.rewards.yaw_wheel_differential_progress.params["command_threshold"] = 0.05
        self.rewards.yaw_wheel_differential_progress.params["max_xy_command"] = 0.20
        self.rewards.yaw_wheel_differential_progress.params["max_yaw_rate"] = 0.8
        self.rewards.yaw_wheel_differential_progress.params["target_wheel_diff"] = 4.0
        self.rewards.yaw_wheel_differential_progress.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.yaw_wheel_velocity_alignment.weight = 0
        self.rewards.yaw_wheel_velocity_alignment.params["command_threshold"] = 0.05
        self.rewards.yaw_wheel_velocity_alignment.params["max_xy_command"] = 0.20
        self.rewards.yaw_wheel_velocity_alignment.params["target_wheel_diff"] = 4.0
        self.rewards.yaw_wheel_velocity_alignment.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.commanded_base_height_below_target.weight = -2.0
        self.rewards.commanded_base_height_below_target.params["target_height"] = 0.56
        self.rewards.commanded_base_height_below_target.params["height_margin"] = 0.06
        self.rewards.commanded_base_height_below_target.params["command_threshold"] = 0.06
        self.rewards.commanded_joint_posture_l2.weight = -0.45
        self.rewards.commanded_joint_posture_l2.params["command_threshold"] = 0.06
        self.rewards.commanded_joint_posture_l2.params["straight_command_only"] = True
        self.rewards.commanded_joint_posture_l2.params["max_abs_yaw_command"] = 0.08
        self.rewards.commanded_joint_posture_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.front_joint_posture_l2.weight = -0.35
        self.rewards.front_joint_posture_l2.params["command_threshold"] = 0.05
        self.rewards.front_joint_posture_l2.params["straight_command_only"] = True
        self.rewards.front_joint_posture_l2.params["max_abs_yaw_command"] = 0.08
        self.rewards.front_joint_posture_l2.params["asset_cfg"].joint_names = [
            "fl_hipy_joint", "fl_knee_joint", "fr_hipy_joint", "fr_knee_joint",
        ]
        self.rewards.yaw_turn_joint_posture_l2.weight = -0.15
        self.rewards.yaw_turn_joint_posture_l2.params["command_threshold"] = 0.06
        self.rewards.yaw_turn_joint_posture_l2.params["yaw_command_only"] = True
        self.rewards.yaw_turn_joint_posture_l2.params["max_xy_command"] = 0.16
        self.rewards.yaw_turn_joint_posture_l2.params["asset_cfg"].joint_names = ".*_hipx_joint"
        self.rewards.joint_deviation_hipx_l1.weight = -0.55
        self.rewards.joint_pos_penalty.weight = -0.40
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = -0.15
        self.rewards.feet_slide.params["command_name"] = "base_velocity"
        self.rewards.feet_slide.params["yaw_command_threshold"] = 0.06
        self.rewards.feet_slide.params["max_xy_command"] = 0.12
        self.rewards.feet_slide.params["yaw_slide_scale"] = 1.0
        self.rewards.feet_air_time.weight = 0.10
        self.rewards.feet_air_time.params["threshold"] = 0.15
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.stuck_with_command.weight = -4.0
        self.rewards.stuck_with_command.params["command_threshold"] = 0.08
        self.rewards.stuck_with_command.params["velocity_threshold"] = 0.08
        self.rewards.yaw_stuck_with_command.weight = -5.0
        self.rewards.yaw_stuck_with_command.params["command_threshold"] = 0.05
        self.rewards.yaw_stuck_with_command.params["yaw_velocity_threshold"] = 0.08
        self.rewards.wheel_spin_when_stuck.weight = 0
        self.rewards.wheel_spin_with_lateral_contact.weight = 0
        self.rewards.commanded_motion_progress.weight = 0
        self.rewards.stair_upward_progress.weight = 0
        self.rewards.upward_without_forward_progress.weight = 0
        self.rewards.vertical_bounce_without_progress.weight = 0
        self.rewards.wheel_spin_without_progress.weight = 0
        self.rewards.wheel_lateral_edge_contact.weight = 0
        self.rewards.wheel_clearance_on_command.weight = 0
        # Unitree/legged_gym-style gait shaping: keep this broad enough to be
        # learnable, then let air-time and anti-slide terms bias real stepping.
        self.rewards.feet_gait.weight = 0.10
        self.rewards.feet_gait.params["std"] = 0.50
        self.rewards.feet_gait.params["command_threshold"] = 0.06
        self.rewards.feet_gait.params["velocity_threshold"] = 0.10
        self.rewards.feet_gait.params["max_err"] = 0.30
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.feet_gait.params["yaw_command_only"] = True
        self.rewards.feet_gait.params["max_xy_command"] = 0.12
        self.rewards.upward.weight = 0

        self.rewards.stand_still.params["command_threshold"] = 0.03
        self.rewards.joint_pos_penalty.params["command_threshold"] = 0.03
        self.rewards.wheel_vel_penalty.params["command_threshold"] = 0.03
        self.rewards.wheel_spin_when_stuck.params["command_threshold"] = 0.08

        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.7, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.35, 0.75)

        # Keyboard play uses arrows for x/y translation and Z/X for yaw.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-0.8, 0.8)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JK04FlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class JK04FlatYawEnvCfg(JK04FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # A focused yaw stage for keyboard X/Z testing.  Keep the first yaw
        # stage slow enough that stepping can beat the easy grounded pivot turn.
        # JK04's early yaw lifts are short; use a lower air-time threshold than
        # the mature 0.50 s gait setting so short exploratory lifts are rewarded.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.45, 0.45)
        self.commands.base_velocity.ranges.heading = (-0.45, 0.45)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.rewards.track_lin_vel_xy_exp.weight = 0.9
        self.rewards.track_ang_vel_z_exp.weight = 2.2
        self.rewards.yaw_command_progress.weight = 0.38
        self.rewards.yaw_command_progress.params["command_threshold"] = 0.05
        self.rewards.yaw_command_progress.params["max_yaw_rate"] = 0.45
        self.rewards.yaw_wheel_differential_progress.weight = 0
        self.rewards.yaw_wheel_differential_progress.params["command_threshold"] = 0.05
        self.rewards.yaw_wheel_differential_progress.params["max_xy_command"] = 1.20
        self.rewards.yaw_wheel_differential_progress.params["max_yaw_rate"] = 0.45
        self.rewards.yaw_wheel_differential_progress.params["target_wheel_diff"] = 4.0
        self.rewards.yaw_wheel_differential_progress.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.yaw_wheel_velocity_alignment.weight = 0
        self.rewards.yaw_wheel_velocity_alignment.params["command_threshold"] = 0.05
        self.rewards.yaw_wheel_velocity_alignment.params["max_xy_command"] = 1.20
        self.rewards.yaw_wheel_velocity_alignment.params["target_wheel_diff"] = 4.0
        self.rewards.yaw_wheel_velocity_alignment.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.yaw_stuck_with_command.weight = -3.0
        self.rewards.yaw_stuck_with_command.params["command_threshold"] = 0.05
        self.rewards.yaw_stuck_with_command.params["yaw_velocity_threshold"] = 0.06
        self.actions.joint_pos.scale = {
            ".*_hipx_joint": 0.035,
            ".*_hipy_joint": 0.30,
            ".*_knee_joint": 0.30,
        }

        # v50: do not add more reward functions. Use existing posture terms to
        # stop the policy from solving yaw by crouching wide and shuffling.
        self.rewards.commanded_base_height_below_target.weight = -0.35
        self.rewards.commanded_base_height_below_target.params["target_height"] = 0.56
        self.rewards.commanded_base_height_below_target.params["height_margin"] = 0.08
        self.rewards.commanded_base_height_below_target.params["command_threshold"] = 0.05
        self.rewards.commanded_base_height_below_target.params["straight_command_only"] = False
        self.rewards.commanded_joint_posture_l2.weight = 0
        self.rewards.front_joint_posture_l2.weight = 0
        self.rewards.stuck_with_command.weight = 0
        self.rewards.feet_slide.weight = -0.11
        self.rewards.feet_slide.params["command_name"] = "base_velocity"
        self.rewards.feet_slide.params["yaw_command_threshold"] = 0.05
        self.rewards.feet_slide.params["max_xy_command"] = 0.12
        self.rewards.feet_slide.params["yaw_slide_scale"] = 1.0
        self.rewards.feet_gait.weight = 0.30
        self.rewards.feet_gait.params["std"] = 0.70
        self.rewards.feet_gait.params["max_err"] = 0.20
        self.rewards.joint_deviation_hipx_l1.weight = -0.60
        self.rewards.joint_pos_penalty.weight = -0.30
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.yaw_turn_joint_posture_l2.weight = 0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.18
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -0.25
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.yaw_feet_air_time_positive.weight = 0.14
        self.rewards.yaw_feet_air_time_positive.params["threshold"] = 0.12
        self.rewards.yaw_feet_air_time_positive.params["command_threshold"] = 0.05
        self.rewards.yaw_feet_air_time_positive.params["max_xy_command"] = 0.12
        self.rewards.yaw_feet_air_time_positive.params["min_contact_feet"] = 2
        self.rewards.yaw_feet_air_time_positive.params["max_air_feet"] = 2
        self.rewards.yaw_feet_air_time_positive.params["sensor_cfg"].body_names = [self.foot_link_name]

        if self.__class__.__name__ == "JK04FlatYawEnvCfg":
            self.disable_zero_weight_rewards()
