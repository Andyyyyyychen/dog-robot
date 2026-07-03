# JK04 Flat Yaw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a registered JK04 flat-yaw Isaac Lab task so `RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0` can pass zero/random-action checks and start PPO training.

**Architecture:** Reuse the proven JK03 wheel-legged task structure and add a focused JK04 variant instead of changing shared locomotion code. JK04 gets its own asset config, RobotLab asset data directory, flat-yaw task package, and PPO experiment name; existing shared reward functions stay unchanged unless a real JK04 bug appears.

**Tech Stack:** Python, Isaac Lab, Gymnasium task registration, RSL-RL PPO, RobotLab asset configs, URDF mesh assets.

---

## File Structure

- Create: `robot_lab-main/source/robot_lab/robot_lab/assets/jk04.py`
  - Defines `JK04_CFG` with JK04 URDF path, initial pose, and actuator groups.
- Create: `robot_lab-main/source/robot_lab/data/Robots/jk04/jk04_description/`
  - Contains JK04 `urdf/`, `mjcf/`, `meshes/`, `config/`, `launch/`, `xacro/`, `package.xml`, and `CMakeLists.txt`.
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/`
  - Contains task registration and JK04 flat-yaw configs.
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/`
  - Contains RSL-RL and CUSRL config stubs matching the local package pattern.
- Modify: `README.md`
  - Add the JK04 flat-yaw train command and the pre-training verification order.
- Modify: `JK03_CHANGELOG.md`
  - Add a dated entry explaining that JK04 was added without changing JK03 protected data or terrain curriculum.

## Task 1: Add JK04 Asset Data

**Files:**
- Create directory: `robot_lab-main/source/robot_lab/data/Robots/jk04/jk04_description/`
- Copy from: `jk04_description/`

- [ ] **Step 1: Copy JK04 description into RobotLab asset data**

Run:

```bash
mkdir -p robot_lab-main/source/robot_lab/data/Robots/jk04
cp -R jk04_description robot_lab-main/source/robot_lab/data/Robots/jk04/
```

Expected: `robot_lab-main/source/robot_lab/data/Robots/jk04/jk04_description/urdf/jk04_description.urdf` exists.

- [ ] **Step 2: Verify XML files still parse**

Run:

```bash
python3 -c 'import xml.etree.ElementTree as ET; ET.parse("robot_lab-main/source/robot_lab/data/Robots/jk04/jk04_description/urdf/jk04_description.urdf"); ET.parse("robot_lab-main/source/robot_lab/data/Robots/jk04/jk04_description/mjcf/jk04_description.xml"); print("jk04 xml ok")'
```

Expected: `jk04 xml ok`.

- [ ] **Step 3: Commit asset data**

Run:

```bash
git add robot_lab-main/source/robot_lab/data/Robots/jk04
git commit -m "Add JK04 robot description assets"
```

Expected: a commit containing only JK04 asset data.

## Task 2: Add JK04 Asset Config

**Files:**
- Create: `robot_lab-main/source/robot_lab/robot_lab/assets/jk04.py`

- [ ] **Step 1: Write JK04 asset config**

Create `robot_lab-main/source/robot_lab/robot_lab/assets/jk04.py`:

```python
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


JK04_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/jk04/jk04_description/urdf/jk04_description.urdf",
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
        pos=(0.0, 0.0, 0.56),
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
            effort_limit=54.0,
            saturation_effort=54.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
        ),
    },
)
```

- [ ] **Step 2: Compile asset config**

Run:

```bash
python3 -m py_compile robot_lab-main/source/robot_lab/robot_lab/assets/jk04.py
```

Expected: command exits with status 0.

- [ ] **Step 3: Commit asset config**

Run:

```bash
git add robot_lab-main/source/robot_lab/robot_lab/assets/jk04.py
git commit -m "Add JK04 Isaac Lab asset config"
```

Expected: a commit containing only `jk04.py`.

## Task 3: Add JK04 Flat-Yaw Task Package

**Files:**
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/__init__.py`
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/rough_env_cfg.py`
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/flat_env_cfg.py`

- [ ] **Step 1: Copy JK03 task package as JK04 starting point**

Run:

```bash
mkdir -p robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04
cp robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/__init__.py robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/__init__.py
cp robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/rough_env_cfg.py
cp robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/flat_env_cfg.py
```

Expected: the three JK04 files exist.

- [ ] **Step 2: Replace JK03 names with JK04 in `__init__.py`**

Edit `jk04/__init__.py` so it registers only this first environment:

```python
# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:JK04FlatYawEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:JK04FlatYawPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:JK04FlatYawTrainerCfg",
    },
)
```

- [ ] **Step 3: Replace JK03 asset import and class names in `rough_env_cfg.py`**

Change:

```python
from robot_lab.assets.jk03 import JK03_CFG
```

to:

```python
from robot_lab.assets.jk04 import JK04_CFG
```

Then rename `JK03ActionsCfg`, `JK03RewardsCfg`, and `JK03RoughEnvCfg` to `JK04ActionsCfg`, `JK04RewardsCfg`, and `JK04RoughEnvCfg`, and replace:

```python
self.scene.robot = JK03_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

with:

```python
self.scene.robot = JK04_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

- [ ] **Step 4: Replace JK03 class names and tune first yaw stage in `flat_env_cfg.py`**

Set the import and class names:

```python
from .rough_env_cfg import JK04RoughEnvCfg


@configclass
class JK04FlatEnvCfg(JK04RoughEnvCfg):
    ...


@configclass
class JK04FlatYawEnvCfg(JK04FlatEnvCfg):
    ...
```

Inside `JK04FlatEnvCfg.__post_init__`, set:

```python
self.rewards.base_height_l2.params["target_height"] = 0.56
self.rewards.commanded_base_height_below_target.params["target_height"] = 0.56
```

Inside `JK04FlatYawEnvCfg.__post_init__`, keep the mixed startup yaw structure and set:

```python
self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
self.commands.base_velocity.ranges.ang_vel_z = (-0.45, 0.45)
self.commands.base_velocity.ranges.heading = (-0.45, 0.45)
self.curriculum.command_levels_lin_vel = None
self.curriculum.command_levels_ang_vel = None
self.rewards.track_ang_vel_z_exp.weight = 2.2
self.rewards.yaw_command_progress.weight = 0.38
self.rewards.yaw_wheel_differential_progress.weight = 0
self.rewards.yaw_wheel_velocity_alignment.weight = 0
self.rewards.yaw_stuck_with_command.weight = -3.0
self.rewards.yaw_in_place_xy_drift_penalty.weight = -0.45
self.rewards.feet_slide.weight = -0.11
self.rewards.feet_gait.weight = 0.30
self.rewards.yaw_feet_air_time_positive.weight = 0.14
self.rewards.yaw_inside_hind_step_participation.weight = 0.30
```

- [ ] **Step 5: Compile JK04 env configs**

Run:

```bash
python3 -m py_compile \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/__init__.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/rough_env_cfg.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/flat_env_cfg.py
```

Expected: command exits with status 0.

## Task 4: Add JK04 Agent Configs

**Files:**
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/__init__.py`
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/rsl_rl_ppo_cfg.py`
- Create: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/cusrl_ppo_cfg.py`

- [ ] **Step 1: Copy JK03 agent configs**

Run:

```bash
mkdir -p robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents
cp robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/agents/__init__.py robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/__init__.py
cp robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/agents/rsl_rl_ppo_cfg.py robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/rsl_rl_ppo_cfg.py
cp robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/agents/cusrl_ppo_cfg.py robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/cusrl_ppo_cfg.py
```

- [ ] **Step 2: Rename RSL-RL classes and experiment name**

In `jk04/agents/rsl_rl_ppo_cfg.py`, rename `JK03RoughPPORunnerCfg`, `JK03FlatPPORunnerCfg`, and `JK03FlatYawPPORunnerCfg` to `JK04RoughPPORunnerCfg`, `JK04FlatPPORunnerCfg`, and `JK04FlatYawPPORunnerCfg`.

Set:

```python
self.max_iterations = 5000
self.experiment_name = "jk04_flat_yaw"
self.policy.init_noise_std = 0.28
self.algorithm.entropy_coef = 0.002
```

inside `JK04FlatYawPPORunnerCfg.__post_init__`.

- [ ] **Step 3: Rename CUSRL classes and experiment name**

In `jk04/agents/cusrl_ppo_cfg.py`, replace `JK03` class prefixes with `JK04`, and set the flat-yaw experiment/log name to `jk04_flat_yaw` wherever the JK03 file sets `jk03_flat_yaw`.

- [ ] **Step 4: Compile agent configs**

Run:

```bash
python3 -m py_compile \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/__init__.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/rsl_rl_ppo_cfg.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/cusrl_ppo_cfg.py
```

Expected: command exits with status 0.

## Task 5: Add Documentation And Changelog

**Files:**
- Modify: `README.md`
- Modify: `JK03_CHANGELOG.md`

- [ ] **Step 1: Add JK04 commands to README**

Add this command block near the training section:

```bash
/root/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py --keyword JK04

/root/IsaacLab/isaaclab.sh -p scripts/tools/zero_agent.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0 \
  --headless \
  --num_envs 16

/root/IsaacLab/isaaclab.sh -p scripts/tools/random_agent.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0 \
  --headless \
  --num_envs 16

/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 5000
```

- [ ] **Step 2: Add changelog entry**

Add a dated entry saying:

```text
2026-07-03
- Added JK04 flat-yaw training entry point.
- Added JK04 robot description assets under RobotLab data.
- Reused existing yaw reward functions and did not change JK03 protected asset data or terrain curriculum.
- Validation: local XML parse and Python compile; cloud validation requires list_envs, zero_agent, and random_agent before PPO training.
```

- [ ] **Step 3: Commit docs**

Run:

```bash
git add README.md JK03_CHANGELOG.md
git commit -m "Document JK04 flat yaw training"
```

Expected: a commit containing only docs/changelog updates.

## Task 6: Verify, Push, And Upload

**Files:**
- All files from previous tasks.

- [ ] **Step 1: Run local static verification**

Run:

```bash
python3 -m py_compile \
  robot_lab-main/source/robot_lab/robot_lab/assets/jk04.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/__init__.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/rough_env_cfg.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/flat_env_cfg.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/rsl_rl_ppo_cfg.py \
  robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/agents/cusrl_ppo_cfg.py
```

Expected: command exits with status 0.

- [ ] **Step 2: Push commits**

Run:

```bash
git push origin main
```

Expected: `main -> main`.

- [ ] **Step 3: Upload the committed version to the cloud server**

Run:

```bash
git archive --format=tar.gz -o /tmp/dog_robot_jk04_flat_yaw.tgz HEAD
scp -P 30141 /tmp/dog_robot_jk04_flat_yaw.tgz root@183.147.142.40:/root/dog_robot_jk04_flat_yaw.tgz
ssh -p 30141 root@183.147.142.40 "mkdir -p /root/dog-robot-main && tar -xzf /root/dog_robot_jk04_flat_yaw.tgz -C /root/dog-robot-main && cd /root/dog-robot-main/robot_lab-main && source /opt/conda/etc/profile.d/conda.sh && conda activate isaaclab && python -m pip install -e source/robot_lab"
```

Expected: pip installs `robot_lab` successfully.

- [ ] **Step 4: Verify JK04 task on cloud**

Run:

```bash
ssh -p 30141 root@183.147.142.40 "cd /root/dog-robot-main/robot_lab-main && source /opt/conda/etc/profile.d/conda.sh && conda activate isaaclab && /root/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py --keyword JK04"
```

Expected output contains:

```text
RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0
```

- [ ] **Step 5: Run cloud zero/random checks**

Run zero action:

```bash
ssh -p 30141 root@183.147.142.40 "cd /root/dog-robot-main/robot_lab-main && source /opt/conda/etc/profile.d/conda.sh && conda activate isaaclab && /root/IsaacLab/isaaclab.sh -p scripts/tools/zero_agent.py --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0 --headless --num_envs 16"
```

Run random action:

```bash
ssh -p 30141 root@183.147.142.40 "cd /root/dog-robot-main/robot_lab-main && source /opt/conda/etc/profile.d/conda.sh && conda activate isaaclab && /root/IsaacLab/isaaclab.sh -p scripts/tools/random_agent.py --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0 --headless --num_envs 16"
```

Expected: each command creates the environment instead of raising `gymnasium.error.NameNotFound`.

- [ ] **Step 6: Start PPO only after checks pass**

Run:

```bash
ssh -p 30141 root@183.147.142.40
tmux new -s jk04_yaw
cd /root/dog-robot-main/robot_lab-main
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 5000
```

Expected: training logs write to `logs/rsl_rl/jk04_flat_yaw/`.

## Self-Review

- Spec coverage: asset config, JK04 asset data, flat-yaw environment registration, PPO config, documentation, local verification, cloud verification, and training launch are covered.
- Placeholder scan: no unresolved marker or incomplete instruction remains in implementation steps.
- Type consistency: JK04 class names match the registered entry points: `JK04FlatYawEnvCfg`, `JK04FlatYawPPORunnerCfg`, and `JK04FlatYawTrainerCfg`.
