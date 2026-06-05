#!/usr/bin/env python3
"""Manual MuJoCo demo for the Unitree Go2 without a trained policy.

This script uses the Go2 URDF/meshes already present in robot_lab-main and
drives the joints with a small procedural trot plus PD torques. It is meant for
local visualization and sanity-checking, not for real hardware.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import mujoco
    import numpy as np
except ModuleNotFoundError as exc:
    missing = exc.name or "mujoco"
    print(
        f"Missing Python package: {missing}\n"
        "Install the MuJoCo Python package on your Linux machine first:\n"
        "  python3 -m pip install mujoco\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    REPO_ROOT
    / "robot_lab-main/source/robot_lab/data/Robots/unitree/go2_description/urdf/go2_description.urdf"
)

JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

DEFAULT_DOF_POS = np.array(
    [
        0.00,
        0.80,
        -1.50,
        0.00,
        0.80,
        -1.50,
        0.00,
        0.80,
        -1.50,
        0.00,
        0.80,
        -1.50,
    ],
    dtype=float,
)

TORQUE_LIMITS = np.array([23.5] * 12, dtype=float)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def prepare_urdf(
    source_urdf: Path,
    output_dir: Path,
    floating_base_height: float,
    *,
    keep_visuals: bool,
) -> Path:
    """Rewrite package mesh paths and inject a floating base joint for MuJoCo."""
    if not source_urdf.exists():
        raise FileNotFoundError(f"Go2 URDF not found: {source_urdf}")

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    description_dir = source_urdf.parents[1]

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        prefix = "package://go2_description/"
        if filename.startswith(prefix):
            mesh.set("filename", str(description_dir / filename[len(prefix) :]))

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
        ET.SubElement(floating_joint, "child", {"link": "base"})
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
    output_path = output_dir / f"go2_manual_mujoco_{suffix}.urdf"
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


class ManualGo2Controller:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        speed: float,
        turn: float,
        stand_height: float,
        balance_assist: bool,
        auto_walk: bool,
    ) -> None:
        self.model = model
        self.data = data
        self.command = np.array([clamp(speed, -1.0, 1.0), 0.0, clamp(turn, -1.0, 1.0)])
        self.stand_height = stand_height
        self.balance_assist = balance_assist
        self.motion_enabled = auto_walk or abs(speed) > 1e-4 or abs(turn) > 1e-4
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
        self.target = DEFAULT_DOF_POS.copy()
        self.kp = np.array([70.0] * 12, dtype=float)
        self.kd = np.array([3.0] * 12, dtype=float)

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
            self.data.qpos[qadr] = DEFAULT_DOF_POS[idx]
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.reset_requested = False

    def on_key(self, key: int) -> None:
        if key < 0:
            return
        char = chr(key).lower() if 0 <= key < 256 else ""

        if char == "w":
            self.command[0] = clamp(self.command[0] + 0.1, -1.0, 1.0)
            self.motion_enabled = True
        elif char == "s":
            self.command[0] = clamp(self.command[0] - 0.1, -1.0, 1.0)
            self.motion_enabled = True
        elif char == "a":
            self.command[1] = clamp(self.command[1] + 0.1, -1.0, 1.0)
            self.motion_enabled = True
        elif char == "d":
            self.command[1] = clamp(self.command[1] - 0.1, -1.0, 1.0)
            self.motion_enabled = True
        elif char == "q":
            self.command[2] = clamp(self.command[2] + 0.1, -1.0, 1.0)
            self.motion_enabled = True
        elif char == "e":
            self.command[2] = clamp(self.command[2] - 0.1, -1.0, 1.0)
            self.motion_enabled = True
        elif char == " ":
            self.command[:] = 0.0
            self.motion_enabled = False
        elif char == "1":
            if np.linalg.norm(self.command) < 1e-4:
                self.command[0] = 0.35
            self.motion_enabled = True
        elif char == "0":
            self.command[:] = 0.0
            self.motion_enabled = False
        elif char == "r":
            self.reset_requested = True
        elif char == "h":
            print_help()
        elif key == 256:
            self.exit_requested = True

    def desired_joint_positions(self) -> np.ndarray:
        desired = DEFAULT_DOF_POS.copy()
        command_mag = float(min(1.0, np.linalg.norm(self.command)))
        if not self.motion_enabled and command_mag < 1e-4:
            return desired

        phase_base = 2.0 * math.pi * (1.15 + 0.65 * command_mag) * self.data.time
        leg_names = ("FR", "FL", "RR", "RL")
        for leg_index, leg_name in enumerate(leg_names):
            offset = 0.0 if leg_name in ("FR", "RL") else math.pi
            phase = phase_base + offset
            sin_phase = math.sin(phase)
            cos_phase = math.cos(phase)
            swing = max(0.0, sin_phase)

            side = -1.0 if leg_name[1] == "R" else 1.0
            front = 1.0 if leg_name[0] == "F" else -1.0
            i = leg_index * 3

            stride = 0.22 * self.command[0] * cos_phase
            lift = 0.30 * swing * max(0.25, command_mag)
            lateral = 0.10 * self.command[1] * side
            yaw = 0.10 * self.command[2] * front * side

            desired[i + 0] += lateral + yaw + 0.025 * sin_phase * command_mag
            desired[i + 1] += stride + lift
            desired[i + 2] += -0.55 * lift - 0.30 * stride

        return np.clip(desired, self.lower_limits + 0.02, self.upper_limits - 0.02)

    def apply_balance_assist(self) -> None:
        if not self.balance_assist or self.free_qpos_addr is None or self.free_qvel_addr is None:
            return

        qadr = self.free_qpos_addr
        vadr = self.free_qvel_addr
        z = float(self.data.qpos[qadr + 2])
        vz = float(self.data.qvel[vadr + 2])
        roll, pitch, _ = quat_to_euler_wxyz(self.data.qpos[qadr + 3 : qadr + 7])

        self.data.qfrc_applied[vadr + 2] += self.total_mass * (
            0.85 * 9.81 + 70.0 * (self.stand_height - z) - 7.0 * vz
        )
        self.data.qfrc_applied[vadr + 3] += -70.0 * roll - 3.5 * float(self.data.qvel[vadr + 3])
        self.data.qfrc_applied[vadr + 4] += -70.0 * pitch - 3.5 * float(self.data.qvel[vadr + 4])

    def step(self) -> None:
        if self.reset_requested:
            self.reset()

        self.data.qfrc_applied[:] = 0.0
        desired = self.desired_joint_positions()
        self.target = 0.90 * self.target + 0.10 * desired

        for idx, (qadr, vadr) in enumerate(zip(self.qpos_addr, self.qvel_addr)):
            position_error = self.target[idx] - float(self.data.qpos[qadr])
            velocity_error = -float(self.data.qvel[vadr])
            torque = self.kp[idx] * position_error + self.kd[idx] * velocity_error
            self.data.qfrc_applied[vadr] = clamp(torque, -TORQUE_LIMITS[idx], TORQUE_LIMITS[idx])

        self.apply_balance_assist()

        if self.data.time - self.last_status > 0.25:
            mode = "trot" if self.motion_enabled else "stand"
            print(
                f"\rmode={mode:<5} x={self.command[0]:+.2f} "
                f"y={self.command[1]:+.2f} yaw={self.command[2]:+.2f} "
                f"assist={'on' if self.balance_assist else 'off'}",
                end="",
                flush=True,
            )
            self.last_status = self.data.time


def print_help() -> None:
    print(
        "\nManual Go2 MuJoCo controls:\n"
        "  1      start procedural trot\n"
        "  0      stand still\n"
        "  W/S    forward/backward speed\n"
        "  A/D    left/right bias\n"
        "  Q/E    yaw bias\n"
        "  Space  clear command and stand\n"
        "  R      reset pose\n"
        "  H      show this help\n"
        "  Esc    close viewer\n"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Path to the Go2 URDF.")
    parser.add_argument("--speed", type=float, default=0.35, help="Initial forward command in [-1, 1].")
    parser.add_argument("--turn", type=float, default=0.0, help="Initial yaw command in [-1, 1].")
    parser.add_argument("--height", type=float, default=0.36, help="Initial/assisted base height in meters.")
    parser.add_argument("--no-assist", action="store_true", help="Disable soft upright balance assist.")
    parser.add_argument("--stand", action="store_true", help="Start standing instead of walking.")
    parser.add_argument("--headless", action="store_true", help="Run without the 3D viewer.")
    parser.add_argument("--duration", type=float, default=10.0, help="Headless run duration in seconds.")
    parser.add_argument("--collision-only", action="store_true", help="Ignore visual meshes and show collision geometry only.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="dog_robot_go2_mujoco_") as tmp:
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

        controller = ManualGo2Controller(
            model,
            data,
            speed=0.0 if args.stand else args.speed,
            turn=args.turn,
            stand_height=args.height,
            balance_assist=not args.no_assist,
            auto_walk=not args.stand,
        )

        print_help()
        if args.headless:
            start_time = data.time
            while data.time - start_time < args.duration:
                controller.step()
                mujoco.mj_step(model, data)
            print()
            return 0

        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
            viewer.cam.distance = 2.0
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
