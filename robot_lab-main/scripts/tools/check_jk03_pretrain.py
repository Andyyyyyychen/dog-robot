#!/usr/bin/env python3
"""Pre-training readiness checks for the JK03 robot setup.

The checks in this file are intentionally read-only. They verify that the JK03
training environments are registered, the MuJoCo helper mirrors the JK03 source
parameters, and no protected JK03 source data has local git changes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROBOT_LAB_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROBOT_LAB_ROOT.parent

JK03_ASSET_CFG = ROBOT_LAB_ROOT / "source/robot_lab/robot_lab/assets/jk03.py"
JK03_TASK_DIR = (
    ROBOT_LAB_ROOT
    / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03"
)
JK03_ENV_INIT = JK03_TASK_DIR / "__init__.py"
JK03_ROUGH_ENV = JK03_TASK_DIR / "rough_env_cfg.py"
JK03_FLAT_ENV = JK03_TASK_DIR / "flat_env_cfg.py"
JK03_RSL_RL_CFG = JK03_TASK_DIR / "agents/rsl_rl_ppo_cfg.py"
JK03_CUSRL_CFG = JK03_TASK_DIR / "agents/cusrl_ppo_cfg.py"
JK03_URDF = ROBOT_LAB_ROOT / "source/robot_lab/data/Robots/jk03/jk03_description/urdf/jk03.urdf"
MANUAL_MUJOCO = WORKSPACE_ROOT / "rl_sar-main/scripts/manual_mujoco_jk03.py"

PROTECTED_PATHS = (
    JK03_ASSET_CFG,
    JK03_TASK_DIR,
    ROBOT_LAB_ROOT / "source/robot_lab/data/Robots/jk03",
)

EXPECTED_TASKS = (
    "RobotLab-Isaac-Velocity-Flat-JK03-v0",
    "RobotLab-Isaac-Velocity-Rough-JK03-v0",
)


@dataclass
class CheckResult:
    level: str
    name: str
    detail: str


class CheckSuite:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def pass_(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult("PASS", name, detail))

    def warn(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult("WARN", name, detail))

    def fail(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult("FAIL", name, detail))

    @property
    def failures(self) -> int:
        return sum(result.level == "FAIL" for result in self.results)

    @property
    def warnings(self) -> int:
        return sum(result.level == "WARN" for result in self.results)

    def print_summary(self) -> None:
        width = max(len(result.name) for result in self.results) if self.results else 0
        for result in self.results:
            detail = f" - {result.detail}" if result.detail else ""
            print(f"[{result.level}] {result.name:<{width}}{detail}")
        print()
        print(f"Summary: {self.failures} failure(s), {self.warnings} warning(s), {len(self.results)} check(s).")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_python(path: Path) -> ast.Module:
    return ast.parse(read_text(path), filename=str(path))


def literal(value: ast.AST):
    return ast.literal_eval(value)


def func_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def attr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{attr_name(node.value)}.{node.attr}"
    return ""


def keyword(call: ast.Call, name: str) -> ast.AST | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def top_level_literals(path: Path, names: set[str]) -> dict[str, object]:
    values: dict[str, object] = {}
    tree = parse_python(path)
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = literal(stmt.value)
    return values


def class_literals(path: Path, class_name: str, names: set[str]) -> dict[str, object]:
    values: dict[str, object] = {}
    tree = parse_python(path)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = literal(stmt.value)
    return values


def post_init_literals(path: Path, class_name: str) -> dict[str, object]:
    values: dict[str, object] = {}
    tree = parse_python(path)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "__post_init__":
                continue
            for stmt in ast.walk(method):
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    name = attr_name(target)
                    if name in {
                        "self.actions.joint_pos.scale",
                        "self.actions.joint_vel.scale",
                    }:
                        values[name] = literal(stmt.value)
    return values


def extract_jk03_asset(path: Path) -> dict[str, object]:
    tree = parse_python(path)
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "JK03_CFG" for target in stmt.targets):
            continue
        if not isinstance(stmt.value, ast.Call):
            raise ValueError("JK03_CFG is not an ArticulationCfg call")

        init_state = keyword(stmt.value, "init_state")
        actuators = keyword(stmt.value, "actuators")
        if not isinstance(init_state, ast.Call) or not isinstance(actuators, ast.Dict):
            raise ValueError("JK03_CFG is missing init_state or actuators")

        data: dict[str, object] = {
            "pos": literal(keyword(init_state, "pos")),
            "joint_pos": literal(keyword(init_state, "joint_pos")),
            "actuators": {},
        }
        actuator_data: dict[str, dict[str, object]] = {}
        for key_node, value_node in zip(actuators.keys, actuators.values):
            if key_node is None or not isinstance(value_node, ast.Call):
                continue
            name = literal(key_node)
            actuator_data[name] = {
                "joint_names_expr": literal(keyword(value_node, "joint_names_expr")),
                "effort_limit": literal(keyword(value_node, "effort_limit")),
                "saturation_effort": literal(keyword(value_node, "saturation_effort")),
                "velocity_limit": literal(keyword(value_node, "velocity_limit")),
                "stiffness": literal(keyword(value_node, "stiffness")),
                "damping": literal(keyword(value_node, "damping")),
            }
        data["actuators"] = actuator_data
        return data
    raise ValueError("JK03_CFG assignment not found")


def actuator_group(joint_name: str) -> str:
    if joint_name.endswith("_wheel_joint"):
        return "wheel"
    if joint_name.endswith("_knee_joint"):
        return "knee"
    if joint_name.endswith("_hipx_joint") or joint_name.endswith("_hipy_joint"):
        return "hip"
    raise ValueError(f"Unknown JK03 joint group: {joint_name}")


def expected_joint_positions(joint_names: list[str], joint_pos_cfg: dict[str, float]) -> tuple[float, ...]:
    values: list[float] = []
    for joint_name in joint_names:
        if joint_name.endswith("_hipx_joint"):
            values.append(joint_pos_cfg[".*_hipx_joint"])
        elif joint_name.endswith("_hipy_joint"):
            values.append(joint_pos_cfg[".*_hipy_joint"])
        elif joint_name.endswith("_knee_joint"):
            values.append(joint_pos_cfg[".*_knee_joint"])
        elif joint_name.endswith("_wheel_joint"):
            values.append(joint_pos_cfg[".*_wheel_joint"])
        else:
            raise ValueError(f"Unknown JK03 joint type: {joint_name}")
    return tuple(values)


def expected_actuator_tuple(
    joint_names: list[str], actuators: dict[str, dict[str, float]], field: str
) -> tuple[float, ...]:
    return tuple(float(actuators[actuator_group(joint_name)][field]) for joint_name in joint_names)


def check_required_files(suite: CheckSuite) -> None:
    required = (
        JK03_ASSET_CFG,
        JK03_ENV_INIT,
        JK03_ROUGH_ENV,
        JK03_FLAT_ENV,
        JK03_RSL_RL_CFG,
        JK03_CUSRL_CFG,
        JK03_URDF,
        MANUAL_MUJOCO,
    )
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        suite.fail("required files", "missing: " + ", ".join(missing))
    else:
        suite.pass_("required files", "JK03 configs, URDF, and MuJoCo helper are present")


def check_git_protection(suite: CheckSuite) -> None:
    proc = subprocess.run(
        ["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        suite.warn("git protection", "workspace is not inside a git repository")
        return

    git_root = Path(proc.stdout.strip()).resolve()
    protected = [str(path.resolve().relative_to(git_root)) for path in PROTECTED_PATHS]
    working = subprocess.run(
        ["git", "-C", str(git_root), "diff", "--quiet", "--", *protected],
        check=False,
    )
    staged = subprocess.run(
        ["git", "-C", str(git_root), "diff", "--cached", "--quiet", "--", *protected],
        check=False,
    )
    untracked = subprocess.run(
        ["git", "-C", str(git_root), "ls-files", "--others", "--exclude-standard", "--", *protected],
        capture_output=True,
        text=True,
        check=False,
    )

    dirty: list[str] = []
    if working.returncode == 1:
        dirty.append("working tree diff")
    if staged.returncode == 1:
        dirty.append("staged diff")
    if untracked.stdout.strip():
        dirty.append("untracked files")
    if working.returncode > 1 or staged.returncode > 1 or untracked.returncode != 0:
        suite.warn("git protection", "git command failed while checking protected JK03 paths")
    elif dirty:
        suite.fail("protected JK03 data", "local changes found: " + ", ".join(dirty))
    else:
        suite.pass_("protected JK03 data", "no git changes under JK03 source data/config paths")


def check_environment_registration(suite: CheckSuite) -> None:
    text = read_text(JK03_ENV_INIT)
    missing = [task for task in EXPECTED_TASKS if task not in text]
    entry_points = (
        "flat_env_cfg:JK03FlatEnvCfg",
        "rough_env_cfg:JK03RoughEnvCfg",
        "rsl_rl_ppo_cfg:JK03FlatPPORunnerCfg",
        "rsl_rl_ppo_cfg:JK03RoughPPORunnerCfg",
        "cusrl_ppo_cfg:JK03FlatTrainerCfg",
        "cusrl_ppo_cfg:JK03RoughTrainerCfg",
    )
    missing_entry_points = [item for item in entry_points if item not in text]
    if missing or missing_entry_points:
        suite.fail("Gym registration", "missing task ids or entry points")
    else:
        suite.pass_("Gym registration", ", ".join(EXPECTED_TASKS))


def check_training_configs(suite: CheckSuite) -> None:
    rsl_text = read_text(JK03_RSL_RL_CFG)
    cusrl_text = read_text(JK03_CUSRL_CFG)
    expected_fragments = (
        (rsl_text, "JK03RoughPPORunnerCfg", "max_iterations = 20000"),
        (rsl_text, "JK03FlatPPORunnerCfg", "self.max_iterations = 5000"),
        (rsl_text, "jk03_rough", 'experiment_name = "jk03_rough"'),
        (rsl_text, "jk03_flat", 'self.experiment_name = "jk03_flat"'),
        (cusrl_text, "JK03RoughTrainerCfg", "max_iterations = 20000"),
        (cusrl_text, "JK03FlatTrainerCfg", "max_iterations = 5000"),
        (cusrl_text, "cusrl rough experiment", 'experiment_name = "jk03_rough"'),
        (cusrl_text, "cusrl flat experiment", 'experiment_name = "jk03_flat"'),
    )
    missing = [label for text, label, fragment in expected_fragments if fragment not in text]
    if missing:
        suite.fail("PPO configs", "missing expected config fragments: " + ", ".join(missing))
    else:
        suite.pass_("PPO configs", "rsl_rl and cusrl JK03 Flat/Rough configs are present")


def check_urdf(suite: CheckSuite, joint_names: list[str]) -> None:
    tree = ET.parse(JK03_URDF)
    root = tree.getroot()
    urdf_joints = {joint.attrib.get("name", "") for joint in root.findall("joint")}
    missing_joints = [joint_name for joint_name in joint_names if joint_name not in urdf_joints]
    if missing_joints:
        suite.fail("URDF joints", "missing from URDF: " + ", ".join(missing_joints))
    else:
        suite.pass_("URDF joints", f"{len(joint_names)} configured joints are present in jk03.urdf")

    missing_meshes: list[str] = []
    mesh_count = 0
    description_dir = JK03_URDF.parents[1]
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        if not filename:
            continue
        mesh_count += 1
        if filename.startswith("package://jk03_description/"):
            mesh_path = description_dir / filename.removeprefix("package://jk03_description/")
        else:
            mesh_path = Path(filename)
            if not mesh_path.is_absolute():
                mesh_path = JK03_URDF.parent / mesh_path
        if not mesh_path.exists():
            missing_meshes.append(filename)
    if missing_meshes:
        suite.fail("URDF meshes", "missing mesh files: " + ", ".join(missing_meshes))
    else:
        suite.pass_("URDF meshes", f"{mesh_count} mesh reference(s) resolve on disk")


def check_parameter_mirror(suite: CheckSuite) -> list[str]:
    manual_names = {
        "LEG_JOINT_NAMES",
        "WHEEL_JOINT_NAMES",
        "WHEEL_INDICES",
        "DEFAULT_BASE_HEIGHT",
        "DEFAULT_DOF_POS",
        "KP",
        "KD",
        "TORQUE_LIMITS",
        "VELOCITY_LIMITS",
        "HIPX_ACTION_SCALE",
        "LEG_ACTION_SCALE",
        "WHEEL_ACTION_SCALE",
    }
    manual = top_level_literals(MANUAL_MUJOCO, manual_names)
    rough = class_literals(JK03_ROUGH_ENV, "JK03RoughEnvCfg", {"leg_joint_names", "wheel_joint_names"})
    rough_post = post_init_literals(JK03_ROUGH_ENV, "JK03RoughEnvCfg")
    asset = extract_jk03_asset(JK03_ASSET_CFG)

    leg_joint_names = list(rough["leg_joint_names"])
    wheel_joint_names = list(rough["wheel_joint_names"])
    joint_names = leg_joint_names + wheel_joint_names
    actuators = asset["actuators"]
    joint_pos_cfg = asset["joint_pos"]

    mismatches: list[str] = []
    comparisons = {
        "LEG_JOINT_NAMES": (manual["LEG_JOINT_NAMES"], leg_joint_names),
        "WHEEL_JOINT_NAMES": (manual["WHEEL_JOINT_NAMES"], wheel_joint_names),
        "WHEEL_INDICES": (manual["WHEEL_INDICES"], tuple(range(len(leg_joint_names), len(joint_names)))),
        "DEFAULT_BASE_HEIGHT": (manual["DEFAULT_BASE_HEIGHT"], asset["pos"][2]),
        "DEFAULT_DOF_POS": (manual["DEFAULT_DOF_POS"], expected_joint_positions(joint_names, joint_pos_cfg)),
        "KP": (manual["KP"], expected_actuator_tuple(joint_names, actuators, "stiffness")),
        "KD": (manual["KD"], expected_actuator_tuple(joint_names, actuators, "damping")),
        "TORQUE_LIMITS": (manual["TORQUE_LIMITS"], expected_actuator_tuple(joint_names, actuators, "effort_limit")),
        "VELOCITY_LIMITS": (manual["VELOCITY_LIMITS"], expected_actuator_tuple(joint_names, actuators, "velocity_limit")),
        "HIPX_ACTION_SCALE": (manual["HIPX_ACTION_SCALE"], rough_post["self.actions.joint_pos.scale"][".*_hipx_joint"]),
        "LEG_ACTION_SCALE": (manual["LEG_ACTION_SCALE"], rough_post["self.actions.joint_pos.scale"]["^(?!.*_hipx_joint).*"]),
        "WHEEL_ACTION_SCALE": (manual["WHEEL_ACTION_SCALE"], rough_post["self.actions.joint_vel.scale"]),
    }
    for label, (left, right) in comparisons.items():
        if left != right:
            mismatches.append(label)

    expected_actuator_exprs = {
        "hip": [".*_hipx_joint", ".*_hipy_joint"],
        "knee": [".*_knee_joint"],
        "wheel": [".*_wheel_joint"],
    }
    for name, exprs in expected_actuator_exprs.items():
        if actuators[name]["joint_names_expr"] != exprs:
            mismatches.append(f"{name}.joint_names_expr")

    if mismatches:
        suite.fail("MuJoCo parameter mirror", "mismatched values: " + ", ".join(mismatches))
    else:
        suite.pass_("MuJoCo parameter mirror", "manual_mujoco_jk03.py matches JK03 source config values")

    return joint_names


def check_runtime_hints(suite: CheckSuite) -> None:
    if importlib.util.find_spec("isaaclab") is None:
        suite.warn("Isaac Lab runtime", "not importable in this Python; train on a Linux Isaac Lab/Isaac Sim environment")
    else:
        suite.pass_("Isaac Lab runtime", "isaaclab package is importable")

    if shutil.which("nvidia-smi") is None:
        suite.warn("NVIDIA GPU runtime", "nvidia-smi not found here; cloud training needs an NVIDIA GPU runtime")
    else:
        suite.pass_("NVIDIA GPU runtime", "nvidia-smi is available")


def check_mujoco_smoke(suite: CheckSuite, skip: bool, timeout: float) -> None:
    if skip:
        suite.warn("MuJoCo smoke test", "skipped by --skip-mujoco")
        return
    if importlib.util.find_spec("mujoco") is None:
        suite.warn("MuJoCo smoke test", "mujoco package is not installed in this Python")
        return

    cmd = [
        sys.executable,
        str(MANUAL_MUJOCO),
        "--headless",
        "--duration",
        "0.02",
        "--scene",
        "stairs-world",
        "--collision-only",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        suite.fail("MuJoCo smoke test", f"timed out after {timeout:.0f}s")
        return

    if proc.returncode == 0:
        suite.pass_("MuJoCo smoke test", "JK03 loads headlessly with the stairs-world scene")
    else:
        output = "\n".join((proc.stdout + proc.stderr).splitlines()[-4:])
        suite.fail("MuJoCo smoke test", output or f"exit code {proc.returncode}")


def print_hashes() -> None:
    files = [
        JK03_ASSET_CFG,
        JK03_ENV_INIT,
        JK03_ROUGH_ENV,
        JK03_FLAT_ENV,
        JK03_RSL_RL_CFG,
        JK03_CUSRL_CFG,
        JK03_URDF,
    ]
    print("Protected JK03 reference hashes:")
    for path in files:
        print(f"  {sha256_file(path)}  {rel(path)}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check JK03 readiness before RL training.")
    parser.add_argument("--skip-mujoco", action="store_true", help="Skip the optional MuJoCo smoke test.")
    parser.add_argument("--mujoco-timeout", type=float, default=30.0, help="MuJoCo smoke-test timeout in seconds.")
    parser.add_argument("--strict-warnings", action="store_true", help="Exit non-zero when warnings are present.")
    parser.add_argument("--no-hashes", action="store_true", help="Do not print protected JK03 file hashes.")
    args = parser.parse_args()

    suite = CheckSuite()
    check_required_files(suite)
    check_git_protection(suite)
    check_environment_registration(suite)
    check_training_configs(suite)
    joint_names = check_parameter_mirror(suite)
    check_urdf(suite, joint_names)
    check_runtime_hints(suite)
    check_mujoco_smoke(suite, args.skip_mujoco, args.mujoco_timeout)

    if not args.no_hashes:
        print_hashes()
    suite.print_summary()

    if suite.failures:
        return 1
    if args.strict_warnings and suite.warnings:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
