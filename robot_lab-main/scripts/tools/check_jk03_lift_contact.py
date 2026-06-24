#!/usr/bin/env python3
"""Open-loop lift/contact diagnostic for JK03.

This tool does not change rewards or training configuration.  It applies simple
scripted leg actions and reports whether wheel bodies actually lift off the
ground, whether the contact sensor records air time, and whether the expected
JK03 body/action ordering is valid.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from isaaclab.app import AppLauncher


LEG_JOINT_NAMES = {
    "fl": ("fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint"),
    "fr": ("fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint"),
    "hl": ("hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint"),
    "hr": ("hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint"),
}
WHEEL_BODY_NAMES = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


parser = argparse.ArgumentParser(description="Check whether JK03 legs can create real wheel air time.")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-Yaw-JK03-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 env for readable diagnostic output.")
parser.add_argument("--settle_time", type=float, default=0.75, help="Seconds of zero action before each phase.")
parser.add_argument("--phase_time", type=float, default=1.25, help="Seconds to hold each scripted action phase.")
parser.add_argument("--lift_action", type=float, default=1.0, help="Normalized hipy/knee action magnitude to test.")
parser.add_argument("--wheel_action", type=float, default=0.0, help="Optional normalized wheel action during lift tests.")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Contact-force norm threshold for contact state.")
parser.add_argument("--min_lift_height", type=float, default=0.015, help="Body-frame z increase counted as visible lift.")
parser.add_argument("--min_air_time", type=float, default=0.015, help="Air-time counted as real contact-sensor lift.")
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
import isaaclab.utils.math as math_utils  # noqa: E402
import robot_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def configure_diagnostic_env(env_cfg) -> None:
    """Make the loaded task deterministic enough for an open-loop lift test."""
    env_cfg.scene.num_envs = args_cli.num_envs
    if hasattr(env_cfg, "observations"):
        if hasattr(env_cfg.observations.policy, "enable_corruption"):
            env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations.critic, "enable_corruption"):
            env_cfg.observations.critic.enable_corruption = False

    if hasattr(env_cfg, "commands"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

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


def empty_action(env) -> torch.Tensor:
    """Create a zero action tensor that works for vectorized or plain gym spaces."""
    shape = env.action_space.shape
    if len(shape) == 1:
        shape = (env.unwrapped.num_envs, shape[0])
    return torch.zeros(shape, device=env.unwrapped.device)


def find_one_body(entity, body_name: str) -> tuple[int, str]:
    """Find one body by name and return its id and resolved name."""
    body_ids, body_names = entity.find_bodies(body_name)
    if len(body_ids) == 0:
        raise RuntimeError(f"Body '{body_name}' was not found in {entity}.")
    return int(body_ids[0]), str(body_names[0])


def entity_joint_names(robot) -> list[str]:
    """Return articulation joint names across Isaac Lab versions."""
    return list(getattr(robot, "joint_names", None) or getattr(robot.data, "joint_names", []))


def entity_body_names(robot) -> list[str]:
    """Return articulation body names across Isaac Lab versions."""
    return list(getattr(robot, "body_names", None) or getattr(robot.data, "body_names", []))


def build_action_layout(robot, action_dim: int) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Map JK03 joint names to action indices from the live articulation order."""
    joint_names = entity_joint_names(robot)
    joint_index_by_name = {name: index for index, name in enumerate(joint_names)}
    action_index_by_joint = {}
    wheel_action_index_by_leg = {}

    for leg, names in LEG_JOINT_NAMES.items():
        for joint_name in names:
            if joint_name not in joint_index_by_name:
                raise RuntimeError(f"Expected joint '{joint_name}' was not found in robot_joint_names={joint_names}.")
            joint_index = joint_index_by_name[joint_name]
            if joint_index >= action_dim:
                raise RuntimeError(
                    f"Joint '{joint_name}' maps to index {joint_index}, outside action dim {action_dim}."
                )
            action_index_by_joint[joint_name] = joint_index

        wheel_joint_name = f"{leg}_wheel_joint"
        if wheel_joint_name not in joint_index_by_name:
            raise RuntimeError(f"Expected wheel joint '{wheel_joint_name}' was not found in robot_joint_names.")
        wheel_index = joint_index_by_name[wheel_joint_name]
        if wheel_index >= action_dim:
            raise RuntimeError(
                f"Wheel joint '{wheel_joint_name}' maps to index {wheel_index}, outside action dim {action_dim}."
            )
        wheel_action_index_by_leg[leg] = wheel_index

    return action_index_by_joint, wheel_action_index_by_leg, joint_index_by_name


def body_frame_wheel_z(robot, body_ids: list[int]) -> torch.Tensor:
    """Return body-frame z positions for selected wheel bodies."""
    body_pos_rel_w = robot.data.body_pos_w[:, body_ids, :] - robot.data.root_pos_w[:, :].unsqueeze(1)
    body_pos_b = torch.zeros_like(body_pos_rel_w)
    for index in range(len(body_ids)):
        body_pos_b[:, index, :] = math_utils.quat_apply_inverse(robot.data.root_quat_w, body_pos_rel_w[:, index, :])
    return body_pos_b[:, :, 2]


def contact_state(contact_sensor, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contact booleans and force norms for selected contact-sensor bodies."""
    force_norm = contact_sensor.data.net_forces_w_history[:, :, body_ids, :].norm(dim=-1).max(dim=1)[0]
    return force_norm > args_cli.contact_threshold, force_norm


def build_leg_action(
    env,
    legs: Iterable[str],
    hipy: float,
    knee: float,
    action_index_by_joint: dict[str, int],
    wheel_action_index_by_leg: dict[str, int],
    wheel_pattern: dict[str, float] | None = None,
):
    """Build a JK03 action with selected legs flexed and optional wheel commands."""
    action = empty_action(env)
    if action.shape[-1] < 16:
        raise RuntimeError(f"Expected JK03 action dim >= 16, got {action.shape[-1]}.")

    for leg in legs:
        hipy_joint = LEG_JOINT_NAMES[leg][1]
        knee_joint = LEG_JOINT_NAMES[leg][2]
        action[:, action_index_by_joint[hipy_joint]] = hipy
        action[:, action_index_by_joint[knee_joint]] = knee

    if wheel_pattern is None:
        wheel_pattern = {}
    for leg, value in wheel_pattern.items():
        action[:, wheel_action_index_by_leg[leg]] = value
    return action


def format_values(values: torch.Tensor, digits: int = 4) -> str:
    """Format the first environment values for compact console output."""
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values[0].detach().cpu()) + "]"


def step_and_get_dones(env, action: torch.Tensor) -> torch.Tensor:
    """Step the env and return done flags for both old and new Gym APIs."""
    result = env.step(action)
    if len(result) == 5:
        _, _, terminated, truncated, _ = result
        return torch.logical_or(terminated, truncated)
    if len(result) == 4:
        _, _, dones, _ = result
        return dones
    raise RuntimeError(f"Unexpected env.step() return length: {len(result)}")


def run_phase(
    env,
    phase_name: str,
    legs: tuple[str, ...],
    hipy_action: float,
    knee_action: float,
    wheel_pattern: dict[str, float] | None,
    robot_body_ids: list[int],
    sensor_body_ids: list[int],
    action_index_by_joint: dict[str, int],
    wheel_action_index_by_leg: dict[str, int],
    joint_index_by_name: dict[str, int],
) -> dict[str, float]:
    """Run one isolated lift-action phase and report kinematic/contact response."""
    print(f"[RUN] starting {phase_name}", flush=True)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    zero_action = empty_action(env)
    action = build_leg_action(
        env, legs, hipy_action, knee_action, action_index_by_joint, wheel_action_index_by_leg, wheel_pattern
    )

    settle_steps = max(int(args_cli.settle_time / step_dt(env)), 1)
    phase_steps = max(int(args_cli.phase_time / step_dt(env)), 1)

    with torch.no_grad():
        for _ in range(settle_steps):
            env.step(zero_action)

        start_z = body_frame_wheel_z(robot, robot_body_ids).clone()
        start_joint_pos = robot.data.joint_pos.clone()

        max_z = start_z.clone()
        max_air = torch.zeros_like(start_z)
        max_force = torch.zeros_like(start_z)
        contact_sum = torch.zeros_like(start_z)
        first_air_sum = torch.zeros_like(start_z)
        dones_count = 0

        for _ in range(phase_steps):
            if not simulation_app.is_running():
                break
            dones = step_and_get_dones(env, action)
            dones_count += int(torch.count_nonzero(dones).detach().cpu())
            cur_z = body_frame_wheel_z(robot, robot_body_ids)
            contacts, force_norm = contact_state(contact_sensor, sensor_body_ids)
            first_air = contact_sensor.compute_first_air(env.unwrapped.step_dt)[:, sensor_body_ids]
            air_time = contact_sensor.data.current_air_time[:, sensor_body_ids]

            max_z = torch.maximum(max_z, cur_z)
            max_air = torch.maximum(max_air, air_time)
            max_force = torch.maximum(max_force, force_norm)
            contact_sum += contacts.float()
            first_air_sum += first_air.float()

        end_joint_pos = robot.data.joint_pos.clone()

    z_delta = max_z - start_z
    contact_fraction = contact_sum / max(float(phase_steps), 1.0)
    joint_delta = torch.abs(end_joint_pos - start_joint_pos)
    selected_joint_delta = []
    for leg in legs:
        hipy_joint = LEG_JOINT_NAMES[leg][1]
        knee_joint = LEG_JOINT_NAMES[leg][2]
        selected_joint_delta.extend(
            [joint_delta[:, joint_index_by_name[hipy_joint]], joint_delta[:, joint_index_by_name[knee_joint]]]
        )
    if selected_joint_delta:
        selected_joint_delta_tensor = torch.stack(selected_joint_delta, dim=1)
        max_selected_joint_delta = float(torch.max(selected_joint_delta_tensor).detach().cpu())
    else:
        max_selected_joint_delta = 0.0

    z_pass = torch.max(z_delta) >= args_cli.min_lift_height
    air_pass = torch.max(max_air) >= args_cli.min_air_time
    first_air_count = int(torch.sum(first_air_sum[0]).detach().cpu())
    verdict = "PASS" if bool(z_pass and air_pass) else "FAIL"

    print(
        "[PHASE] "
        f"{phase_name:<22} legs={','.join(legs) or '-':<11} "
        f"hipy={hipy_action:+.2f} knee={knee_action:+.2f} "
        f"z_delta={format_values(z_delta)} "
        f"max_air={format_values(max_air)} "
        f"contact_frac={format_values(contact_fraction, 3)} "
        f"first_air_events={first_air_count} "
        f"max_joint_delta={max_selected_joint_delta:.4f} "
        f"dones={dones_count} verdict={verdict}"
        ,
        flush=True,
    )

    if not bool(z_pass) and max_selected_joint_delta < 0.02:
        print(
            "[HINT] Selected joints barely moved. Check action ordering, action scale, joint limits, or actuator gains.",
            flush=True,
        )
    elif bool(z_pass) and not bool(air_pass):
        print(
            "[HINT] Wheel z increased but air_time stayed low. Check contact body mapping or ground contact threshold.",
            flush=True,
        )
    elif not bool(z_pass) and max_selected_joint_delta >= 0.02:
        print("[HINT] Joints moved but wheel z did not lift. Check lift sign, linkage geometry, or default posture.", flush=True)

    return {
        "max_z_delta": float(torch.max(z_delta).detach().cpu()),
        "max_air_time": float(torch.max(max_air).detach().cpu()),
        "mean_contact_fraction": float(torch.mean(contact_fraction).detach().cpu()),
        "max_joint_delta": max_selected_joint_delta,
        "first_air_events": float(first_air_count),
        "dones": float(dones_count),
    }


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_diagnostic_env(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]

    robot_body_ids = []
    robot_body_names = []
    sensor_body_ids = []
    sensor_body_names = []
    for body_name in WHEEL_BODY_NAMES:
        body_id, resolved_name = find_one_body(robot, body_name)
        robot_body_ids.append(body_id)
        robot_body_names.append(resolved_name)
        sensor_id, resolved_sensor_name = find_one_body(contact_sensor, body_name)
        sensor_body_ids.append(sensor_id)
        sensor_body_names.append(resolved_sensor_name)

    joint_names = entity_joint_names(robot)
    body_names = entity_body_names(robot)
    action_dim = empty_action(env).shape[-1]
    action_index_by_joint, wheel_action_index_by_leg, joint_index_by_name = build_action_layout(robot, action_dim)
    print(f"[INFO] task={args_cli.task}", flush=True)
    print(f"[INFO] action_space={env.action_space}", flush=True)
    print(f"[INFO] control_dt={step_dt(env):.5f}s", flush=True)
    print(f"[INFO] robot_joint_names={joint_names}", flush=True)
    print(f"[INFO] robot_body_names={body_names}", flush=True)
    print(f"[INFO] wheel_body_ids={list(zip(WHEEL_BODY_NAMES, robot_body_ids, robot_body_names))}", flush=True)
    print(f"[INFO] contact_body_ids={list(zip(WHEEL_BODY_NAMES, sensor_body_ids, sensor_body_names))}", flush=True)
    print(f"[INFO] leg_action_index_by_joint={action_index_by_joint}", flush=True)
    print(f"[INFO] wheel_action_index_by_leg={wheel_action_index_by_leg}", flush=True)
    print(f"[INFO] leg_joint_names={LEG_JOINT_NAMES}", flush=True)

    phases = [
        ("all_h+_k+", ("fl", "fr", "hl", "hr"), +args_cli.lift_action, +args_cli.lift_action, None),
        ("all_h+_k-", ("fl", "fr", "hl", "hr"), +args_cli.lift_action, -args_cli.lift_action, None),
        ("all_h-_k+", ("fl", "fr", "hl", "hr"), -args_cli.lift_action, +args_cli.lift_action, None),
        ("all_h-_k-", ("fl", "fr", "hl", "hr"), -args_cli.lift_action, -args_cli.lift_action, None),
        ("front_h+_k+", ("fl", "fr"), +args_cli.lift_action, +args_cli.lift_action, None),
        ("hind_h+_k+", ("hl", "hr"), +args_cli.lift_action, +args_cli.lift_action, None),
        ("diag_fl_hr_h+_k+", ("fl", "hr"), +args_cli.lift_action, +args_cli.lift_action, None),
        ("diag_fr_hl_h+_k+", ("fr", "hl"), +args_cli.lift_action, +args_cli.lift_action, None),
    ]
    if abs(args_cli.wheel_action) > 0.0:
        phases.extend(
            [
                (
                    "yaw_left_plus_lift",
                    ("fl", "fr", "hl", "hr"),
                    +args_cli.lift_action,
                    +args_cli.lift_action,
                    {"fl": args_cli.wheel_action, "fr": -args_cli.wheel_action, "hl": args_cli.wheel_action, "hr": -args_cli.wheel_action},
                ),
                (
                    "yaw_right_plus_lift",
                    ("fl", "fr", "hl", "hr"),
                    +args_cli.lift_action,
                    +args_cli.lift_action,
                    {"fl": -args_cli.wheel_action, "fr": args_cli.wheel_action, "hl": -args_cli.wheel_action, "hr": args_cli.wheel_action},
                ),
            ]
        )

    print(f"[INFO] running {len(phases)} scripted lift phases", flush=True)
    results = [
        run_phase(
            env,
            name,
            legs,
            hipy,
            knee,
            wheel_pattern,
            robot_body_ids,
            sensor_body_ids,
            action_index_by_joint,
            wheel_action_index_by_leg,
            joint_index_by_name,
        )
        for name, legs, hipy, knee, wheel_pattern in phases
    ]
    best = max(results, key=lambda item: (item["max_air_time"], item["max_z_delta"]))
    print("[SUMMARY]", flush=True)
    print(
        "best_phase_metrics "
        f"max_z_delta={best['max_z_delta']:.4f} "
        f"max_air_time={best['max_air_time']:.4f} "
        f"mean_contact_fraction={best['mean_contact_fraction']:.3f} "
        f"max_joint_delta={best['max_joint_delta']:.4f} "
        f"first_air_events={int(best['first_air_events'])} "
        f"dones={int(best['dones'])}",
        flush=True,
    )
    if best["max_air_time"] < args_cli.min_air_time:
        print(
            "FAIL: scripted leg actions did not create measurable wheel air_time. "
            "Do not tune reward weights blindly; first inspect action sign/scale, linkage geometry, contact body mapping, or default posture.",
            flush=True,
        )
    else:
        print(
            "PASS: scripted actions can create wheel air_time. "
            "If PPO still does not lift, focus on reward balance, command distribution, and policy exploration.",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
