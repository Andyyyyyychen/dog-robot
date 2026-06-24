# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.jk03 import JK03_CFG  # isort: skip


@configclass
class JK03ActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=20.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class JK03RewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    stuck_with_command = RewTerm(
        func=mdp.stuck_with_command,
        weight=0.0,
        params={"command_name": "base_velocity", "command_threshold": 0.35, "velocity_threshold": 0.12},
    )

    yaw_stuck_with_command = RewTerm(
        func=mdp.yaw_stuck_with_command,
        weight=0.0,
        params={"command_name": "base_velocity", "command_threshold": 0.10, "yaw_velocity_threshold": 0.06},
    )

    yaw_command_progress = RewTerm(
        func=mdp.yaw_command_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "max_yaw_rate": 0.35,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    yaw_wheel_differential_progress = RewTerm(
        func=mdp.yaw_wheel_differential_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "max_xy_command": 0.18,
            "max_yaw_rate": 0.70,
            "target_wheel_diff": 5.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    yaw_wheel_velocity_alignment = RewTerm(
        func=mdp.yaw_wheel_velocity_alignment,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "max_xy_command": 1.2,
            "target_wheel_diff": 5.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    commanded_base_height_below_target = RewTerm(
        func=mdp.commanded_base_height_below_target,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "target_height": 0.456,
            "height_margin": 0.08,
            "command_threshold": 0.08,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    commanded_joint_posture_l2 = RewTerm(
        func=mdp.commanded_joint_posture_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    front_joint_posture_l2 = RewTerm(
        func=mdp.commanded_joint_posture_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    yaw_turn_joint_posture_l2 = RewTerm(
        func=mdp.commanded_joint_posture_l2,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    wheel_spin_when_stuck = RewTerm(
        func=mdp.wheel_spin_when_stuck,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.35,
            "velocity_threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    wheel_spin_with_lateral_contact = RewTerm(
        func=mdp.wheel_spin_with_lateral_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "base_lateral_force_ratio": 2.0,
            "min_lateral_force_ratio": 1.3,
            "max_lateral_force_ratio": 3.0,
            "vertical_force_eps": 5.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    commanded_motion_progress = RewTerm(
        func=mdp.commanded_motion_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "max_progress": 0.04,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    stair_upward_progress = RewTerm(
        func=mdp.stair_upward_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "max_forward_step": 0.30,
            "max_up_step": 0.09,
            "min_forward_step": 0.08,
            "min_up_step": 0.025,
            "window_steps": 30,
            "forward_weight": 0.0,
            "coupled_weight": 0.85,
            "upward_weight": 0.15,
            "min_forward_fraction": 0.30,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    upward_without_forward_progress = RewTerm(
        func=mdp.upward_without_forward_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "max_up_step": 0.09,
            "min_forward_step": 0.08,
            "window_steps": 30,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    vertical_bounce_without_progress = RewTerm(
        func=mdp.vertical_bounce_without_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "velocity_threshold": 0.12,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    wheel_spin_without_progress = RewTerm(
        func=mdp.wheel_spin_without_progress,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "velocity_threshold": 0.08,
            "wheel_speed_threshold": 2.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
        },
    )

    wheel_lateral_edge_contact = RewTerm(
        func=mdp.wheel_lateral_edge_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "base_lateral_force_ratio": 1.8,
            "min_lateral_force_ratio": 1.2,
            "max_lateral_force_ratio": 2.8,
            "vertical_force_eps": 5.0,
        },
    )

    wheel_clearance_on_command = RewTerm(
        func=mdp.wheel_clearance_on_command,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "min_height": -0.34,
            "target_height": -0.24,
            "tanh_mult": 2.0,
            "forward_velocity_threshold": 0.08,
            "min_progress_scale": 0.35,
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )

@configclass
class JK03RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: JK03ActionsCfg = JK03ActionsCfg()
    rewards: JK03RewardsCfg = JK03RewardsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_wheel"

    # fmt: off
    leg_joint_names = [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    ]
    wheel_joint_names = [
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = JK03_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.max_init_terrain_level = 1
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None and terrain_generator.sub_terrains is not None:
            stair_training_proportions = {
                "random_rough": 0.10,
                "hf_pyramid_slope": 0.05,
                "hf_pyramid_slope_inv": 0.05,
                "boxes": 0.10,
                "pyramid_stairs": 0.65,
                "pyramid_stairs_inv": 0.05,
            }
            for terrain_name, proportion in stair_training_proportions.items():
                if terrain_name in terrain_generator.sub_terrains:
                    terrain_generator.sub_terrains[terrain_name].proportion = proportion
            for terrain_name in ("pyramid_stairs", "pyramid_stairs_inv"):
                if terrain_name not in terrain_generator.sub_terrains:
                    continue
                stair_cfg = terrain_generator.sub_terrains[terrain_name]
                if hasattr(stair_cfg, "step_height_range"):
                    stair_cfg.step_height_range = (0.02, 0.10)
                if hasattr(stair_cfg, "step_width"):
                    stair_cfg.step_width = 0.45
                if hasattr(stair_cfg, "platform_width"):
                    stair_cfg.platform_width = 2.2

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # Keep base velocity visible to the actor for JK03 retraining. The
        # previous policy received commands but learned actions that barely
        # moved the heavy body, so this closes that feedback loop.
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hipx_joint": 0.125, "^(?!.*_hipx_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -0.7
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.flat_orientation_l2.weight = 0
        # Keep the heavy body from solving stairs by crouching into edges.
        self.rewards.base_height_l2.weight = -0.30
        self.rewards.base_height_l2.params["target_height"] = 0.456
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hipx_l1", -0.25, [".*_hipx_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -1.0e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["command_threshold"] = 0.03
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = -0.80
        self.rewards.joint_pos_penalty.params["command_threshold"] = 0.03
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.commanded_joint_posture_l2.weight = -0.25
        self.rewards.commanded_joint_posture_l2.params["command_threshold"] = 0.08
        self.rewards.commanded_joint_posture_l2.params["straight_command_only"] = True
        self.rewards.commanded_joint_posture_l2.params["max_abs_yaw_command"] = 0.08
        self.rewards.commanded_joint_posture_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = -0.08
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.008

        # Contact sensor
        self.rewards.undesired_contacts.weight = -3.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -2.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.contact_forces.params["threshold"] = 500.0
        self.rewards.stuck_with_command.weight = -3.5
        self.rewards.stuck_with_command.params["command_threshold"] = 0.08
        self.rewards.stuck_with_command.params["velocity_threshold"] = 0.08
        self.rewards.yaw_stuck_with_command.weight = -1.5
        self.rewards.yaw_stuck_with_command.params["command_threshold"] = 0.08
        self.rewards.yaw_stuck_with_command.params["yaw_velocity_threshold"] = 0.06
        self.rewards.commanded_base_height_below_target.weight = -0.50
        self.rewards.commanded_base_height_below_target.params["target_height"] = 0.456
        self.rewards.commanded_base_height_below_target.params["height_margin"] = 0.10
        self.rewards.commanded_base_height_below_target.params["command_threshold"] = 0.08
        self.rewards.commanded_base_height_below_target.params["straight_command_only"] = True
        self.rewards.commanded_base_height_below_target.params["max_abs_yaw_command"] = 0.08
        self.rewards.wheel_spin_when_stuck.weight = -1.0e-4
        self.rewards.wheel_spin_when_stuck.params["command_threshold"] = 0.08
        self.rewards.wheel_spin_when_stuck.params["velocity_threshold"] = 0.08
        self.rewards.wheel_spin_when_stuck.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_spin_with_lateral_contact.weight = -2.0e-4
        self.rewards.wheel_spin_with_lateral_contact.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_spin_with_lateral_contact.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.commanded_motion_progress.weight = 1.0
        self.rewards.stair_upward_progress.weight = 2.0
        self.rewards.upward_without_forward_progress.weight = -0.8
        self.rewards.vertical_bounce_without_progress.weight = -0.25
        self.rewards.wheel_spin_without_progress.weight = -0.03
        self.rewards.wheel_spin_without_progress.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.wheel_lateral_edge_contact.weight = -0.06
        self.rewards.wheel_lateral_edge_contact.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_clearance_on_command.weight = 0.8
        self.rewards.wheel_clearance_on_command.params["asset_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.35
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # Encourage wheel clearance relative to the body while moving, so the
        # policy learns to lift over stair edges instead of scraping through.
        self.rewards.feet_height_body.weight = -0.08
        self.rewards.feet_height_body.params["target_height"] = -0.30
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["command_threshold"] = 0.08
        self.rewards.feet_gait.params["velocity_threshold"] = 0.15
        self.rewards.feet_gait.params["max_err"] = 0.25
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "JK03RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact.params["threshold"] = 20.0

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.5, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.5, 1.0)

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.rel_standing_envs = 0.0
        # Include a conservative reverse range so keyboard/play and PPO both
        # learn what a negative x-velocity command means.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.35, 0.9)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-0.5, 0.5)
