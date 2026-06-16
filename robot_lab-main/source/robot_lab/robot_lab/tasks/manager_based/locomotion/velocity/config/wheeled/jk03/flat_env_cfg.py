# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import JK03RoughEnvCfg


@configclass
class JK03FlatEnvCfg(JK03RoughEnvCfg):
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
        # Keep the JK03 initial pose unchanged, but remove rough-terrain early
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
            ".*_hipy_joint": 0.17,
            ".*_knee_joint": 0.17,
        }
        self.actions.joint_vel.scale = 10.0

        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.15
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.456
        self.rewards.track_lin_vel_xy_exp.weight = 8.0
        self.rewards.track_ang_vel_z_exp.weight = 3.2
        self.rewards.yaw_command_progress.weight = 1.2
        self.rewards.yaw_command_progress.params["command_threshold"] = 0.06
        self.rewards.yaw_command_progress.params["max_yaw_rate"] = 0.70
        self.rewards.commanded_base_height_below_target.weight = -3.5
        self.rewards.commanded_base_height_below_target.params["target_height"] = 0.456
        self.rewards.commanded_base_height_below_target.params["height_margin"] = 0.06
        self.rewards.commanded_base_height_below_target.params["command_threshold"] = 0.06
        self.rewards.commanded_joint_posture_l2.weight = -0.8
        self.rewards.commanded_joint_posture_l2.params["command_threshold"] = 0.06
        self.rewards.commanded_joint_posture_l2.params["straight_command_only"] = True
        self.rewards.commanded_joint_posture_l2.params["max_abs_yaw_command"] = 0.08
        self.rewards.commanded_joint_posture_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.front_joint_posture_l2.weight = -0.70
        self.rewards.front_joint_posture_l2.params["command_threshold"] = 0.05
        self.rewards.front_joint_posture_l2.params["straight_command_only"] = True
        self.rewards.front_joint_posture_l2.params["max_abs_yaw_command"] = 0.08
        self.rewards.front_joint_posture_l2.params["asset_cfg"].joint_names = [
            "fl_hipy_joint", "fl_knee_joint", "fr_hipy_joint", "fr_knee_joint",
        ]
        self.rewards.yaw_turn_joint_posture_l2.weight = -1.0
        self.rewards.yaw_turn_joint_posture_l2.params["command_threshold"] = 0.06
        self.rewards.yaw_turn_joint_posture_l2.params["yaw_command_only"] = True
        self.rewards.yaw_turn_joint_posture_l2.params["max_xy_command"] = 0.18
        self.rewards.yaw_turn_joint_posture_l2.params["asset_cfg"].joint_names = ".*_hipx_joint"
        self.rewards.joint_deviation_hipx_l1.weight = -0.35
        self.rewards.joint_pos_penalty.weight = -1.1
        self.rewards.action_rate_l2.weight = -0.003
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = -0.26
        self.rewards.feet_height_body.weight = 0
        self.rewards.stuck_with_command.weight = -4.0
        self.rewards.stuck_with_command.params["command_threshold"] = 0.08
        self.rewards.stuck_with_command.params["velocity_threshold"] = 0.08
        self.rewards.yaw_stuck_with_command.weight = -3.0
        self.rewards.yaw_stuck_with_command.params["command_threshold"] = 0.08
        self.rewards.yaw_stuck_with_command.params["yaw_velocity_threshold"] = 0.06
        self.rewards.wheel_spin_when_stuck.weight = 0
        self.rewards.wheel_spin_with_lateral_contact.weight = 0
        self.rewards.commanded_motion_progress.weight = 0
        self.rewards.stair_upward_progress.weight = 0
        self.rewards.upward_without_forward_progress.weight = 0
        self.rewards.vertical_bounce_without_progress.weight = 0
        self.rewards.wheel_spin_without_progress.weight = 0
        self.rewards.wheel_lateral_edge_contact.weight = 0
        self.rewards.wheel_clearance_on_command.weight = 0
        self.rewards.feet_gait.weight = 0.90
        self.rewards.feet_gait.params["std"] = 0.55
        self.rewards.feet_gait.params["command_threshold"] = 0.06
        self.rewards.feet_gait.params["velocity_threshold"] = 0.10
        self.rewards.feet_gait.params["max_err"] = 0.30
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.feet_gait.params["yaw_command_only"] = True
        self.rewards.feet_gait.params["max_xy_command"] = 0.12
        self.rewards.yaw_turn_feet_clearance.weight = 2.40
        self.rewards.yaw_turn_feet_clearance.params["command_threshold"] = 0.06
        self.rewards.yaw_turn_feet_clearance.params["max_xy_command"] = 0.12
        self.rewards.yaw_turn_feet_clearance.params["target_height"] = -0.255
        self.rewards.yaw_turn_feet_clearance.params["min_air_time"] = 0.025
        self.rewards.yaw_turn_feet_clearance.params["max_air_time"] = 0.25
        self.rewards.yaw_turn_feet_clearance.params["tanh_mult"] = 5.0
        self.rewards.yaw_turn_feet_clearance.params["min_base_height"] = 0.435
        self.rewards.yaw_turn_feet_clearance.params["base_height_margin"] = 0.03
        self.rewards.yaw_turn_feet_clearance.params["synced_feet_pair_names"] = (
            ("fl_wheel", "hr_wheel"),
            ("fr_wheel", "hl_wheel"),
        )
        self.rewards.yaw_turn_feet_clearance.params["min_contact_time"] = 0.015
        self.rewards.yaw_turn_feet_clearance.params["diagonal_pair_weight"] = 0.95
        self.rewards.yaw_turn_feet_clearance.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.yaw_turn_feet_clearance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.yaw_turn_diagonal_step.weight = 1.80
        self.rewards.yaw_turn_diagonal_step.params["command_threshold"] = 0.06
        self.rewards.yaw_turn_diagonal_step.params["max_xy_command"] = 0.12
        self.rewards.yaw_turn_diagonal_step.params["synced_feet_pair_names"] = (
            ("fl_wheel", "hr_wheel"),
            ("fr_wheel", "hl_wheel"),
        )
        self.rewards.yaw_turn_diagonal_step.params["min_air_time"] = 0.025
        self.rewards.yaw_turn_diagonal_step.params["min_contact_time"] = 0.015
        self.rewards.yaw_turn_diagonal_step.params["phase_balance_weight"] = 0.20
        self.rewards.yaw_turn_diagonal_step.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0

        self.rewards.stand_still.params["command_threshold"] = 0.03
        self.rewards.joint_pos_penalty.params["command_threshold"] = 0.03
        self.rewards.wheel_vel_penalty.params["command_threshold"] = 0.03
        self.rewards.wheel_spin_when_stuck.params["command_threshold"] = 0.08

        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.7, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.7, 1.0)

        # Keyboard play uses arrows for x/y translation and Z/X for yaw.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.45, 0.45)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.9, 0.9)
        self.commands.base_velocity.ranges.heading = (-0.9, 0.9)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JK03FlatEnvCfg":
            self.disable_zero_weight_rewards()
