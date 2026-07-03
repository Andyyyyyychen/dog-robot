# JK04 Flat Yaw Training Design

Date: 2026-07-03

## Goal

Start JK04 reinforcement learning from a focused flat-ground yaw task. The first training target is stable near-in-place turning on a plane: the robot should follow commanded yaw rate, keep low xy drift, stay upright, and avoid collapsing into a crouched wheel-scrubbing solution.

The recommended strategy is a two-stage reward schedule:

1. Mixed startup yaw: let the wheels help enough for the policy to start turning, while the main rewards still prefer stable supported stepping.
2. Stricter stepping yaw: once the policy can turn reliably, reduce wheel shortcuts and increase penalties/rewards that favor short supported foot or wheel lifts during yaw.

## Current Context

The repository already has a mature JK03 wheel-legged training path in `robot_lab-main`, including:

- `robot_lab.assets.jk03.JK03_CFG`
- `wheeled/jk03` flat, flat-yaw, and rough environment configs
- yaw-specific reward functions in `velocity/mdp/rewards.py`
- RSL-RL PPO configs for JK03 flat yaw

JK04 currently exists as `jk04_description/` with URDF, MJCF, meshes, Gazebo/xacro, and control files. It is not yet registered as a RobotLab training asset or Gym environment.

## Scope

Implement a JK04 flat-yaw training entry point by copying the proven JK03 structure and changing only what is needed for JK04:

- Add a JK04 asset config that points to the JK04 URDF.
- Add `wheeled/jk04` task configs for flat yaw training.
- Register `RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0`.
- Add PPO config names and experiment name for JK04.
- Reuse existing yaw reward functions unless a JK04-specific bug appears.
- Keep JK03 protected data and terrain curriculum untouched.

Do not start rough terrain, stairs, or forward/lateral locomotion training in this step.

## Reward Design

The first implementation should create one flat-yaw environment with mixed startup weights. These values are meant as the first run, not final truth.

Command distribution:

- `lin_vel_x = 0`
- `lin_vel_y = 0`
- `ang_vel_z = (-0.45, 0.45)`
- no linear or angular curriculum in the first yaw stage

Primary positive rewards:

- strong `track_ang_vel_z_exp`
- moderate `yaw_command_progress`
- small to moderate `feet_gait` for diagonal stepping rhythm during yaw
- small `yaw_feet_air_time_positive` to reward short supported lifts
- moderate `yaw_inside_hind_step_participation` so turning does not become only front-leg or wheel motion

Primary penalties:

- `yaw_in_place_xy_drift_penalty` to keep turns near-in-place
- `flat_orientation_l2`, `ang_vel_xy_l2`, and base-height penalties for stability
- `feet_slide` for yaw slide control
- `joint_deviation_hipx_l1`, `joint_pos_penalty`, `action_rate_l2`, and leg joint acceleration penalties to prevent violent or crouched solutions
- `yaw_stuck_with_command` to punish receiving a yaw command but barely rotating

Wheel shortcut handling:

- In mixed startup yaw, do not heavily penalize wheel differential at first.
- Keep wheel-differential positive rewards disabled, so the policy is not explicitly paid for wheel-only yaw.
- After the policy can turn, create a stricter variant by increasing slide/posture discipline and reducing any behavior that looks like wheel scrubbing without supported stepping.

## JK04 Parameters To Confirm In Code

The initial code should infer as much as possible from JK04's URDF/MJCF:

- joint names follow the JK03 pattern: `fl/fr/hl/hr` plus `hipx`, `hipy`, `knee`, and `wheel`
- feet/contact bodies are the wheel links: `.*_wheel`
- base link is `base_link`
- nominal standing height should start at `0.56 m`, derived from the JK04 MJCF default geometry where the base is at `0.60 m`, the wheel centers sit about `0.50 m` below the base, and the wheel radius is about `0.06 m`
- actuator limits should start from JK04 MJCF values: hip/knee around 96/128 Nm and wheel around 54 Nm, then tune after zero/random-action checks

The `0.56 m` height is a first-run assumption, not a tuned result. Verify it with zero-action Isaac Lab runs before long PPO training.

## Testing And Verification

Local static checks:

- Python compile for changed task and asset files.
- XML parse for JK04 URDF/MJCF.
- grep/list environment registration to confirm the JK04 task is discoverable.

Isaac Lab checks on the training machine:

- install/update `robot_lab-main`
- list environments with keyword `JK04`
- run a small zero-action test on `RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0`
- run a small random-action test
- start short headless PPO smoke run before full training

Success for the first real run:

- policy turns in both yaw directions on flat ground
- yaw rate follows the command roughly
- xy drift remains small
- robot does not fall, crouch excessively, or solve only by wheel scrubbing

## Risks

- JK04 URDF mesh/package paths may need relocation under RobotLab assets before Isaac Lab can load them.
- Initial base height may be wrong and cause early falls or crouching.
- If JK04 inertial data differs from JK03 enough, copied JK03 reward weights may need immediate retuning.
- The policy may discover wheel-only yaw if the first stage is left unchanged too long.

## Non-Goals

- No terrain curriculum changes.
- No JK03 reward or protected asset edits.
- No stair or rough-terrain JK04 training in this step.
- No sim-to-real deployment assumptions yet.
