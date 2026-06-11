# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


JK03_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/jk03/jk03_description/urdf/jk03.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.43),
        joint_pos={
            ".*_hipx_joint": 0.0,
            ".*_hipy_joint": 0.9,
            ".*_knee_joint": -1.33,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hipx_joint", ".*_hipy_joint"],
            effort_limit=96.0,
            saturation_effort=96.0,
            velocity_limit=140.0,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=128.0,
            saturation_effort=128.0,
            velocity_limit=100.0,
            stiffness=80.0,
            damping=2.0,
            friction=0.0,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=96.0,
            saturation_effort=96.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
        ),
    },
)
