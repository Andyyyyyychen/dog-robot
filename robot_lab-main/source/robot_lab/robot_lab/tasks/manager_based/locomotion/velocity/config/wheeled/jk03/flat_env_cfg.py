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

        # Flat pretraining should first learn clean forward and turning commands.
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

        self.rewards.track_lin_vel_xy_exp.weight = 8.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        self.rewards.yaw_command_progress.weight = 1.5
        self.rewards.yaw_command_progress.params["command_threshold"] = 0.06
        self.rewards.yaw_command_progress.params["max_yaw_rate"] = 0.35
        self.rewards.commanded_base_height_below_target.weight = -1.2
        self.rewards.commanded_base_height_below_target.params["target_height"] = 0.43
        self.rewards.commanded_base_height_below_target.params["height_margin"] = 0.08
        self.rewards.commanded_base_height_below_target.params["command_threshold"] = 0.06
        self.rewards.joint_pos_penalty.weight = -0.45
        self.rewards.action_rate_l2.weight = -0.003
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_height_body.weight = 0
        self.rewards.stuck_with_command.weight = -4.0
        self.rewards.stuck_with_command.params["command_threshold"] = 0.08
        self.rewards.stuck_with_command.params["velocity_threshold"] = 0.08
        self.rewards.yaw_stuck_with_command.weight = -3.0
        self.rewards.yaw_stuck_with_command.params["command_threshold"] = 0.08
        self.rewards.yaw_stuck_with_command.params["yaw_velocity_threshold"] = 0.06
        self.rewards.wheel_spin_when_stuck.weight = 0
        self.rewards.wheel_spin_with_lateral_contact.weight = 0
        self.rewards.commanded_motion_progress.weight = 1.0
        self.rewards.stair_upward_progress.weight = 0
        self.rewards.wheel_spin_without_progress.weight = -0.02
        self.rewards.wheel_lateral_edge_contact.weight = 0

        self.rewards.stand_still.params["command_threshold"] = 0.03
        self.rewards.joint_pos_penalty.params["command_threshold"] = 0.03
        self.rewards.wheel_vel_penalty.params["command_threshold"] = 0.03
        self.rewards.wheel_spin_when_stuck.params["command_threshold"] = 0.08

        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.5, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.5, 1.0)

        # Keyboard play sends direct yaw-rate commands, so flat pretraining
        # should learn direct yaw-rate tracking instead of heading-target turns.
        self.commands.base_velocity.heading_command = False

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JK03FlatEnvCfg":
            self.disable_zero_weight_rewards()
