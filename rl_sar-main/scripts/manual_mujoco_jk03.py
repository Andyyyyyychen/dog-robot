#!/usr/bin/env python3
"""Manual MuJoCo demo for JK03 using the robot_lab JK03 parameters.

This script loads the JK03 URDF that already exists under robot_lab-main and
uses the joint order, initial pose, actuator gains, and effort limits from the
JK03 robot_lab config files. It only writes a temporary MuJoCo-ready URDF; it
does not modify the JK03 source URDF or any JK03 config parameter.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


np = None
mujoco = None
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    REPO_ROOT
    / "robot_lab-main/source/robot_lab/data/Robots/jk03/jk03_description/urdf/jk03.urdf"
)

# From robot_lab .../config/wheeled/jk03/rough_env_cfg.py.
LEG_JOINT_NAMES = [
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
]
WHEEL_JOINT_NAMES = [
    "fl_wheel_joint",
    "fr_wheel_joint",
    "hl_wheel_joint",
    "hr_wheel_joint",
]
JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
WHEEL_INDICES = (12, 13, 14, 15)

# From robot_lab/source/robot_lab/robot_lab/assets/jk03.py.
DEFAULT_BASE_HEIGHT = 0.52
DEFAULT_DOF_POS = (
    0.0,
    0.0,
    -1.2,
    0.0,
    0.0,
    -1.2,
    0.0,
    0.0,
    -1.2,
    0.0,
    0.0,
    -1.2,
    0.0,
    0.0,
    0.0,
    0.0,
)
KP = (
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    80.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
KD = (
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    0.6,
    0.6,
    0.6,
    0.6,
)
TORQUE_LIMITS = (
    96.0,
    96.0,
    128.0,
    96.0,
    96.0,
    128.0,
    96.0,
    96.0,
    128.0,
    96.0,
    96.0,
    128.0,
    96.0,
    96.0,
    96.0,
    96.0,
)
VELOCITY_LIMITS = (
    140.0,
    140.0,
    100.0,
    140.0,
    140.0,
    100.0,
    140.0,
    140.0,
    100.0,
    140.0,
    140.0,
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
)

# From rough_env_cfg.py action scales.
HIPX_ACTION_SCALE = 0.125
LEG_ACTION_SCALE = 0.25
WHEEL_ACTION_SCALE = 5.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def import_mujoco():
    try:
        import mujoco as mujoco_module
    except ModuleNotFoundError as exc:
        print(
            "Missing Python package: mujoco\n"
            "Install the MuJoCo Python package on your machine first:\n"
            "  python3 -m pip install mujoco\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return mujoco_module


def import_numpy():
    try:
        import numpy as numpy_module
    except ModuleNotFoundError as exc:
        print(
            "Missing Python package: numpy\n"
            "Install the MuJoCo Python package on your machine first; it includes NumPy as a dependency:\n"
            "  python3 -m pip install mujoco\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return numpy_module


def quat_to_euler_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def resolve_mesh_filename(source_urdf: Path, filename: str) -> str:
    if not filename:
        return filename
    if filename.startswith("package://jk03_description/"):
        description_dir = source_urdf.parents[1]
        return str(description_dir / filename[len("package://jk03_description/") :])
    path = Path(filename)
    if path.is_absolute():
        return str(path)
    return str((source_urdf.parent / path).resolve())


def prepare_urdf(
    source_urdf: Path,
    output_dir: Path,
    floating_base_height: float,
    *,
    keep_visuals: bool,
) -> Path:
    """Rewrite mesh paths and add a floating base joint for MuJoCo."""
    if not source_urdf.exists():
        raise FileNotFoundError(f"JK03 URDF not found: {source_urdf}")

    tree = ET.parse(source_urdf)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        mesh.set("filename", resolve_mesh_filename(source_urdf, mesh.attrib.get("filename", "")))

    # MuJoCo does not need the ROS transmission blocks, and removing them only
    # affects this temporary copy.
    for transmission in list(root.findall("transmission")):
        root.remove(transmission)

    for visual in root.findall(".//visual"):
        for material in list(visual.findall("material")):
            visual.remove(material)

    if root.find("link[@name='world']") is None:
        root.insert(0, ET.Element("link", {"name": "world"}))

    if root.find("joint[@name='floating_base']") is None:
        floating_joint = ET.Element("joint", {"name": "floating_base", "type": "floating"})
        ET.SubElement(
            floating_joint,
            "origin",
            {"xyz": f"0 0 {floating_base_height:.4f}", "rpy": "0 0 0"},
        )
        ET.SubElement(floating_joint, "parent", {"link": "world"})
        ET.SubElement(floating_joint, "child", {"link": "base_link"})
        root.insert(1, floating_joint)

    mujoco_node = root.find("mujoco")
    if mujoco_node is None:
        mujoco_node = ET.SubElement(root, "mujoco")
    compiler = mujoco_node.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_node, "compiler")
    compiler.set("discardvisual", "false" if keep_visuals else "true")
    compiler.set("fusestatic", "true")

    suffix = "visual" if keep_visuals else "collision"
    output_path = output_dir / f"jk03_manual_mujoco_{suffix}.urdf"
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


class ManualJK03Controller:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        speed: float,
        turn: float,
        stand_height: float,
        balance_assist: bool,
        auto_drive: bool,
    ) -> None:
        self.model = model
        self.data = data
        self.command = np.array([clamp(speed, -1.0, 1.0), 0.0, clamp(turn, -1.0, 1.0)])
        self.stand_height = stand_height
        self.balance_assist = balance_assist
        self.drive_enabled = auto_drive or abs(speed) > 1e-4 or abs(turn) > 1e-4
        self.reset_requested = False
        self.exit_requested = False
        self.last_status = 0.0

        self.joint_ids = []
        self.qpos_addr = []
        self.qvel_addr = []
        self.lower_limits = []
        self.upper_limits = []
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"Joint not found in MuJoCo model: {name}")
            self.joint_ids.append(joint_id)
            self.qpos_addr.append(int(model.jnt_qposadr[joint_id]))
            self.qvel_addr.append(int(model.jnt_dofadr[joint_id]))
            if model.jnt_limited[joint_id]:
                self.lower_limits.append(float(model.jnt_range[joint_id][0]))
                self.upper_limits.append(float(model.jnt_range[joint_id][1]))
            else:
                self.lower_limits.append(-math.inf)
                self.upper_limits.append(math.inf)

        self.qpos_addr = np.array(self.qpos_addr, dtype=int)
        self.qvel_addr = np.array(self.qvel_addr, dtype=int)
        self.lower_limits = np.array(self.lower_limits, dtype=float)
        self.upper_limits = np.array(self.upper_limits, dtype=float)
        self.wheel_indices = np.array(WHEEL_INDICES, dtype=int)
        self.default_dof_pos = np.array(DEFAULT_DOF_POS, dtype=float)
        self.kp = np.array(KP, dtype=float)
        self.kd = np.array(KD, dtype=float)
        self.torque_limits = np.array(TORQUE_LIMITS, dtype=float)
        self.velocity_limits = np.array(VELOCITY_LIMITS, dtype=float)
        self.target_pos = self.default_dof_pos.copy()
        self.target_vel = np.zeros(len(JOINT_NAMES), dtype=float)

        self.free_qpos_addr = None
        self.free_qvel_addr = None
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                self.free_qpos_addr = int(model.jnt_qposadr[joint_id])
                self.free_qvel_addr = int(model.jnt_dofadr[joint_id])
                break

        self.total_mass = float(np.sum(model.body_mass))
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self.free_qpos_addr is not None:
            qadr = self.free_qpos_addr
            self.data.qpos[qadr : qadr + 3] = np.array([0.0, 0.0, self.stand_height])
            self.data.qpos[qadr + 3 : qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        for idx, qadr in enumerate(self.qpos_addr):
            self.data.qpos[qadr] = self.default_dof_pos[idx]
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.target_pos = self.default_dof_pos.copy()
        self.target_vel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.reset_requested = False

    def on_key(self, key: int) -> None:
        if key < 0:
            return
        char = chr(key).lower() if 0 <= key < 256 else ""

        if char == "w":
            self.command[0] = clamp(self.command[0] + 0.1, -1.0, 1.0)
            self.drive_enabled = True
        elif char == "s":
            self.command[0] = clamp(self.command[0] - 0.1, -1.0, 1.0)
            self.drive_enabled = True
        elif char == "a":
            self.command[1] = clamp(self.command[1] + 0.1, -1.0, 1.0)
            self.drive_enabled = True
        elif char == "d":
            self.command[1] = clamp(self.command[1] - 0.1, -1.0, 1.0)
            self.drive_enabled = True
        elif char == "q":
            self.command[2] = clamp(self.command[2] + 0.1, -1.0, 1.0)
            self.drive_enabled = True
        elif char == "e":
            self.command[2] = clamp(self.command[2] - 0.1, -1.0, 1.0)
            self.drive_enabled = True
        elif char == " ":
            self.command[:] = 0.0
            self.drive_enabled = False
        elif char == "1":
            if np.linalg.norm(self.command) < 1e-4:
                self.command[0] = 0.35
            self.drive_enabled = True
        elif char == "0":
            self.command[:] = 0.0
            self.drive_enabled = False
        elif char == "r":
            self.reset_requested = True
        elif char == "h":
            print_help()
        elif key == 256:
            self.exit_requested = True

    def desired_leg_positions(self) -> np.ndarray:
        desired = self.default_dof_pos.copy()
        if not self.drive_enabled:
            return desired

        forward = self.command[0]
        lateral = self.command[1]
        yaw = self.command[2]

        phase = 2.0 * math.pi * 0.7 * self.data.time
        for leg_offset, leg_name in enumerate(("fl", "fr", "hl", "hr")):
            idx = leg_offset * 3
            side = 1.0 if leg_name.endswith("l") else -1.0
            front = 1.0 if leg_name.startswith("f") else -1.0

            # Small posture bias using the same action scales as JK03 training.
            desired[idx + 0] += HIPX_ACTION_SCALE * (0.2 * lateral * side + 0.15 * yaw * front * side)
            desired[idx + 1] += LEG_ACTION_SCALE * 0.08 * forward * math.sin(phase + leg_offset * math.pi)
            desired[idx + 2] += LEG_ACTION_SCALE * 0.05 * forward * math.cos(phase + leg_offset * math.pi)

        return np.clip(desired, self.lower_limits + 0.02, self.upper_limits - 0.02)

    def desired_wheel_velocities(self) -> np.ndarray:
        wheel_vel = np.zeros(4, dtype=float)
        if not self.drive_enabled:
            return wheel_vel

        forward = self.command[0]
        lateral = self.command[1]
        yaw = self.command[2]
        for index, name in enumerate(WHEEL_JOINT_NAMES):
            side = 1.0 if name.startswith(("fl", "hl")) else -1.0
            wheel_vel[index] = WHEEL_ACTION_SCALE * (forward - side * yaw + 0.25 * lateral * side)
        return np.clip(wheel_vel, -self.velocity_limits[self.wheel_indices], self.velocity_limits[self.wheel_indices])

    def apply_balance_assist(self) -> None:
        if not self.balance_assist or self.free_qpos_addr is None or self.free_qvel_addr is None:
            return

        qadr = self.free_qpos_addr
        vadr = self.free_qvel_addr
        z = float(self.data.qpos[qadr + 2])
        vz = float(self.data.qvel[vadr + 2])
        roll, pitch, _ = quat_to_euler_wxyz(self.data.qpos[qadr + 3 : qadr + 7])

        self.data.qfrc_applied[vadr + 2] += self.total_mass * (
            0.85 * 9.81 + 80.0 * (self.stand_height - z) - 8.0 * vz
        )
        self.data.qfrc_applied[vadr + 3] += -80.0 * roll - 4.0 * float(self.data.qvel[vadr + 3])
        self.data.qfrc_applied[vadr + 4] += -80.0 * pitch - 4.0 * float(self.data.qvel[vadr + 4])

    def step(self) -> None:
        if self.reset_requested:
            self.reset()

        self.data.qfrc_applied[:] = 0.0
        desired = self.desired_leg_positions()
        self.target_pos = 0.9 * self.target_pos + 0.1 * desired
        self.target_vel[self.wheel_indices] = self.desired_wheel_velocities()

        for idx, (qadr, vadr) in enumerate(zip(self.qpos_addr, self.qvel_addr)):
            if idx in WHEEL_INDICES:
                velocity_error = self.target_vel[idx] - float(self.data.qvel[vadr])
                torque = self.kd[idx] * velocity_error
            else:
                position_error = self.target_pos[idx] - float(self.data.qpos[qadr])
                velocity_error = -float(self.data.qvel[vadr])
                torque = self.kp[idx] * position_error + self.kd[idx] * velocity_error
            self.data.qfrc_applied[vadr] = clamp(torque, -self.torque_limits[idx], self.torque_limits[idx])

        self.apply_balance_assist()

        if self.data.time - self.last_status > 0.25:
            mode = "drive" if self.drive_enabled else "stand"
            wheel = self.target_vel[self.wheel_indices]
            print(
                f"\rmode={mode:<5} x={self.command[0]:+.2f} "
                f"y={self.command[1]:+.2f} yaw={self.command[2]:+.2f} "
                f"wheel=[{wheel[0]:+.1f},{wheel[1]:+.1f},{wheel[2]:+.1f},{wheel[3]:+.1f}] "
                f"assist={'on' if self.balance_assist else 'off'}",
                end="",
                flush=True,
            )
            self.last_status = self.data.time


def print_help() -> None:
    print(
        "\nManual JK03 MuJoCo controls:\n"
        "  1      start wheel drive\n"
        "  0      stand still\n"
        "  W/S    forward/backward wheel command\n"
        "  A/D    left/right posture and wheel bias\n"
        "  Q/E    yaw bias\n"
        "  Space  clear command and stand\n"
        "  R      reset pose\n"
        "  H      show this help\n"
        "  Esc    close viewer\n"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Path to the JK03 URDF.")
    parser.add_argument("--speed", type=float, default=0.35, help="Initial forward command in [-1, 1].")
    parser.add_argument("--turn", type=float, default=0.0, help="Initial yaw command in [-1, 1].")
    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_BASE_HEIGHT,
        help="Initial/assisted base height in meters.",
    )
    parser.add_argument("--no-assist", action="store_true", help="Disable soft upright balance assist.")
    parser.add_argument("--stand", action="store_true", help="Start standing instead of driving.")
    parser.add_argument("--headless", action="store_true", help="Run without the 3D viewer.")
    parser.add_argument("--duration", type=float, default=10.0, help="Headless run duration in seconds.")
    parser.add_argument(
        "--collision-only",
        action="store_true",
        help="Ignore visual meshes and show collision geometry only.",
    )
    return parser


def main() -> int:
    global mujoco, np
    args = build_arg_parser().parse_args()
    np = import_numpy()
    mujoco = import_mujoco()
    with tempfile.TemporaryDirectory(prefix="dog_robot_jk03_mujoco_") as tmp:
        tmp_dir = Path(tmp)
        keep_visuals = not args.collision_only
        prepared_urdf = prepare_urdf(args.urdf, tmp_dir, args.height, keep_visuals=keep_visuals)
        try:
            model = mujoco.MjModel.from_xml_path(str(prepared_urdf))
        except Exception as exc:
            if args.collision_only:
                raise
            print(
                "Visual mesh loading failed; retrying with collision geometry only.\n"
                f"Original MuJoCo error: {exc}",
                file=sys.stderr,
            )
            prepared_urdf = prepare_urdf(args.urdf, tmp_dir, args.height, keep_visuals=False)
            model = mujoco.MjModel.from_xml_path(str(prepared_urdf))
        model.opt.timestep = 0.002
        data = mujoco.MjData(model)

        controller = ManualJK03Controller(
            model,
            data,
            speed=0.0 if args.stand else args.speed,
            turn=args.turn,
            stand_height=args.height,
            balance_assist=not args.no_assist,
            auto_drive=not args.stand,
        )

        print_help()
        if args.headless:
            start_time = data.time
            while data.time - start_time < args.duration:
                controller.step()
                mujoco.mj_step(model, data)
            print()
            return 0

        viewer_module = importlib.import_module("mujoco.viewer")

        with viewer_module.launch_passive(model, data, key_callback=controller.on_key) as viewer:
            viewer.cam.distance = 2.4
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -18
            while viewer.is_running() and not controller.exit_requested:
                step_started = time.time()
                controller.step()
                mujoco.mj_step(model, data)
                viewer.sync()
                sleep_time = model.opt.timestep - (time.time() - step_started)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
