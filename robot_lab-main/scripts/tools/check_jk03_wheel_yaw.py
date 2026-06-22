#!/usr/bin/env python3
"""Open-loop wheel yaw diagnostic for JK03.

This tool bypasses the trained policy and applies direct wheel-velocity actions
to verify whether the JK03 model can physically rotate from left/right wheel
differential commands. It is read-only: no JK03 parameters, URDF, rewards, or
terrain curriculum are modified.
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check whether JK03 can yaw from open-loop wheel actions.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-JK03-v0", help="Task name to load.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Keep this at 1 for diagnosis.")
parser.add_argument(
    "--duration",
    type=float,
    default=4.0,
    help="Seconds to apply each wheel test phase after settling.",
)
parser.add_argument(
    "--settle_time",
    type=float,
    default=0.75,
    help="Seconds of zero action before each phase.",
)
parser.add_argument(
    "--wheel_action",
    type=float,
    default=1.0,
    help="Normalized wheel action magnitude. With JK03 wheel scale 5.0, 1.0 means about 5 rad/s target.",
)
parser.add_argument(
    "--leg_action",
    type=float,
    default=0.0,
    help="Constant normalized leg action applied to all leg channels. Default keeps the default pose.",
)
parser.add_argument(
    "--min_yaw_deg",
    type=float,
    default=8.0,
    help="Minimum absolute yaw change in degrees to consider a differential phase physically effective.",
)
parser.add_argument(
    "--min_abs_wz",
    type=float,
    default=0.05,
    help="Minimum absolute mean body yaw rate to consider a differential phase physically effective.",
)
parser.add_argument(
    "--include_forward",
    action="store_true",
    help="Also test all wheels with the same sign to diagnose wheel direction conventions.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import robot_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def yaw_from_quat_wxyz(quat: torch.Tensor) -> float:
    """Return yaw angle in radians from an Isaac Lab wxyz quaternion."""
    w, x, y, z = quat.tolist()
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def configure_diagnostic_env(env_cfg) -> None:
    """Make the loaded task deterministic enough for an open-loop yaw test."""
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg, "observations"):
        if hasattr(env_cfg.observations.policy, "enable_corruption"):
            env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations.critic, "enable_corruption"):
            env_cfg.observations.critic.enable_corruption = False

    if hasattr(env_cfg, "events"):
        for event_name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_others",
            "randomize_rigid_body_mass",
            "randomize_com_positions",
            "randomize_apply_external_force_torque",
            "randomize_push_robot",
            "push_robot",
            "randomize_reset_joints",
            "randomize_actuator_gains",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

        if hasattr(env_cfg.events, "randomize_reset_base") and env_cfg.events.randomize_reset_base is not None:
            env_cfg.events.randomize_reset_base.params = {
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
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

    terrain = getattr(env_cfg.scene, "terrain", None)
    if terrain is not None and getattr(terrain, "terrain_generator", None) is not None:
        terrain.max_init_terrain_level = 0
        terrain.terrain_generator.curriculum = False
        terrain.terrain_generator.num_rows = 1
        terrain.terrain_generator.num_cols = 1


def step_dt(env) -> float:
    """Best-effort control step duration."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "step_dt"):
        return float(unwrapped.step_dt)
    if hasattr(unwrapped, "physics_dt"):
        return float(unwrapped.physics_dt)
    return float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)


def build_action(env, wheel_pattern: tuple[float, float, float, float]) -> torch.Tensor:
    """Build an action with fixed leg channels and four wheel channels at the end."""
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action_dim = action.shape[-1]
    if action_dim < 16:
        raise RuntimeError(f"Expected at least 16 JK03 action channels, got {action_dim}.")

    wheel_start = action_dim - 4
    if wheel_start > 0:
        action[:, :wheel_start] = args_cli.leg_action
    action[:, wheel_start:] = torch.tensor(wheel_pattern, device=env.unwrapped.device) * args_cli.wheel_action
    return action


def run_phase(env, phase_name: str, wheel_pattern: tuple[float, float, float, float], steps: int, settle_steps: int):
    """Run one isolated wheel-action phase and report body response."""
    env.reset()
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    action = build_action(env, wheel_pattern)

    with torch.no_grad():
        for _ in range(settle_steps):
            env.step(zero_action)

        robot = env.unwrapped.scene["robot"]
        yaw_start = yaw_from_quat_wxyz(robot.data.root_quat_w[0].detach().cpu())

        wz_samples: list[float] = []
        vx_samples: list[float] = []
        vy_samples: list[float] = []
        for _ in range(steps):
            if not simulation_app.is_running():
                break
            env.step(action)
            wz_samples.append(float(robot.data.root_ang_vel_b[0, 2].detach().cpu()))
            vx_samples.append(float(robot.data.root_lin_vel_b[0, 0].detach().cpu()))
            vy_samples.append(float(robot.data.root_lin_vel_b[0, 1].detach().cpu()))

        yaw_end = yaw_from_quat_wxyz(robot.data.root_quat_w[0].detach().cpu())

    yaw_delta = wrap_angle(yaw_end - yaw_start)
    mean_wz = sum(wz_samples) / max(len(wz_samples), 1)
    mean_vx = sum(vx_samples) / max(len(vx_samples), 1)
    mean_vy = sum(vy_samples) / max(len(vy_samples), 1)
    passed = abs(math.degrees(yaw_delta)) >= args_cli.min_yaw_deg or abs(mean_wz) >= args_cli.min_abs_wz

    print(
        "[RESULT] "
        f"{phase_name:<12} "
        f"wheel_action={[round(v, 3) for v in action[0, -4:].detach().cpu().tolist()]} "
        f"yaw_delta={math.degrees(yaw_delta): .2f} deg "
        f"mean_wz={mean_wz: .4f} rad/s "
        f"mean_vx={mean_vx: .4f} m/s "
        f"mean_vy={mean_vy: .4f} m/s "
        f"verdict={'PASS' if passed else 'FAIL'}"
    )
    return phase_name, yaw_delta, mean_wz, passed


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_diagnostic_env(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] action_space={env.action_space}")
    print("[INFO] wheel channel assumption: last 4 actions are [FL, FR, RL, RR] wheel velocity commands.")
    print("[INFO] yaw_left uses left wheels positive and right wheels negative; yaw_right reverses that.")

    dt = step_dt(env)
    steps = max(int(args_cli.duration / dt), 1)
    settle_steps = max(int(args_cli.settle_time / dt), 1)
    print(f"[INFO] control_dt={dt:.5f}s phase_steps={steps} settle_steps={settle_steps}")

    phases: list[tuple[str, tuple[float, float, float, float]]] = [
        ("yaw_left", (1.0, -1.0, 1.0, -1.0)),
        ("yaw_right", (-1.0, 1.0, -1.0, 1.0)),
    ]
    if args_cli.include_forward:
        phases.extend(
            [
                ("all_positive", (1.0, 1.0, 1.0, 1.0)),
                ("all_negative", (-1.0, -1.0, -1.0, -1.0)),
            ]
        )

    results = [run_phase(env, name, pattern, steps, settle_steps) for name, pattern in phases]

    yaw_results = {name: (yaw_delta, mean_wz, passed) for name, yaw_delta, mean_wz, passed in results}
    left = yaw_results.get("yaw_left")
    right = yaw_results.get("yaw_right")
    print("[SUMMARY]")
    if left is not None and right is not None:
        left_yaw, left_wz, left_pass = left
        right_yaw, right_wz, right_pass = right
        if not left_pass and not right_pass:
            print(
                "FAIL: open-loop wheel differential barely rotates the base. "
                "Check wheel joint direction, friction/contact, action scale, or wheel drive strength before PPO tuning."
            )
        elif math.copysign(1.0, left_yaw if abs(left_yaw) > 1e-4 else left_wz) == math.copysign(
            1.0, right_yaw if abs(right_yaw) > 1e-4 else right_wz
        ):
            print(
                "WARN: yaw_left and yaw_right rotate in the same direction. "
                "Wheel sign/order is likely wrong for the assumed [FL, FR, RL, RR] mapping."
            )
        else:
            print(
                "PASS: open-loop wheel differential can rotate the base in opposite directions. "
                "If the trained policy still cannot yaw, focus on command mapping, reward balance, and PPO training."
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
