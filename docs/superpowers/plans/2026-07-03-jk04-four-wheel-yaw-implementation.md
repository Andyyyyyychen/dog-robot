# JK04 Four-Wheel Yaw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the JK04 flat-yaw task reward four-wheel differential yaw instead of rear-wheel-dominant yaw.

**Architecture:** Add a JK04-local reward function in the JK04 wheeled config module and register it in `JK04RewardsCfg`. Tune only `JK04FlatYawEnvCfg` reward weights so this behavior is isolated to the yaw stage.

**Tech Stack:** IsaacLab manager-based RL config, RSL-RL PPO, Python stdlib tests using AST/source checks.

---

### Task 1: Lock Expected JK04 Yaw Reward Config

**Files:**
- Create: `tests/test_jk04_four_wheel_yaw_config.py`
- Read: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/rough_env_cfg.py`
- Read: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/flat_env_cfg.py`

- [ ] **Step 1: Write the failing test**

Create a stdlib test that checks for `yaw_front_rear_wheel_participation`, nonzero weights for the new term, nonzero wheel yaw alignment/progress rewards, and reduced pure `yaw_command_progress`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 tests/test_jk04_four_wheel_yaw_config.py`

Expected: failure because the new reward term is not present and wheel yaw reward weights are zero.

### Task 2: Add JK04-Local Four-Wheel Yaw Reward

**Files:**
- Modify: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/rough_env_cfg.py`
- Modify: `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04/flat_env_cfg.py`

- [ ] **Step 1: Add the JK04-local reward function**

Add `yaw_front_rear_wheel_participation` next to the JK04 reward config. It must read the four wheel joint velocities in preserve order, gate on near-in-place yaw commands, multiply by signed yaw progress, and reward balanced front/rear wheel motion.

- [ ] **Step 2: Register the reward term**

Add `yaw_front_rear_wheel_participation = RewTerm(...)` to `JK04RewardsCfg` with default weight `0.0`.

- [ ] **Step 3: Tune `JK04FlatYawEnvCfg`**

Set nonzero weights for:

- `yaw_front_rear_wheel_participation`
- `yaw_wheel_velocity_alignment`
- `yaw_wheel_differential_progress`

Reduce `yaw_command_progress`, disable yaw stepping rewards, and keep flat posture penalties moderate.

- [ ] **Step 4: Run tests**

Run: `python3 tests/test_jk04_four_wheel_yaw_config.py`

Expected: pass.

### Task 3: Verify, Commit, Push, Upload

**Files:**
- Commit only JK04 yaw files, the new test, and these docs.

- [ ] **Step 1: Run syntax checks**

Run: `python3 -m py_compile <changed Python files>`

- [ ] **Step 2: Commit and push**

Commit the scoped changes and push to `origin/main`.

- [ ] **Step 3: Upload to cloud and install**

Upload the archive to `/root/dog-robot-main`, reinstall `robot_lab`, and write the uploaded commit marker.

- [ ] **Step 4: Cloud validation**

Run a one-iteration RSL-RL training command for `RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0` and confirm it exits successfully.
