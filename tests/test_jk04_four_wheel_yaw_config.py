"""Static checks for JK04 flat-yaw four-wheel participation rewards."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JK04_DIR = ROOT / "robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk04"
ROUGH_CFG = JK04_DIR / "rough_env_cfg.py"
FLAT_CFG = JK04_DIR / "flat_env_cfg.py"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def assigned_number(text: str, dotted_name: str) -> float:
    pattern = re.compile(rf"{re.escape(dotted_name)}\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing assignment for {dotted_name}")
    return float(match.group(1))


def class_source(text: str, class_name: str) -> str:
    start = text.index(f"class {class_name}")
    next_class = text.find("\nclass ", start + 1)
    if next_class == -1:
        return text[start:]
    return text[start:next_class]


class JK04FourWheelYawConfigTest(unittest.TestCase):
    def test_rough_cfg_defines_front_rear_participation_reward(self) -> None:
        tree = ast.parse(source(ROUGH_CFG))
        function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn("yaw_front_rear_wheel_participation", function_names)

        text = source(ROUGH_CFG)
        self.assertIn("yaw_front_rear_wheel_participation = RewTerm(", text)
        self.assertIn("func=yaw_front_rear_wheel_participation", text)
        self.assertIn('"max_yaw_rate"', text)
        self.assertIn('"target_wheel_speed"', text)

    def test_flat_yaw_enables_wheel_participation_rewards(self) -> None:
        text = class_source(source(FLAT_CFG), "JK04FlatYawEnvCfg")
        self.assertGreater(
            assigned_number(text, "self.rewards.yaw_front_rear_wheel_participation.weight"),
            0.0,
        )
        self.assertGreater(
            assigned_number(text, "self.rewards.yaw_wheel_velocity_alignment.weight"),
            0.0,
        )
        self.assertGreater(
            assigned_number(text, "self.rewards.yaw_wheel_differential_progress.weight"),
            0.0,
        )
        self.assertLessEqual(
            assigned_number(text, "self.rewards.yaw_command_progress.weight"),
            0.25,
        )

    def test_flat_yaw_disables_step_based_yaw_shaping(self) -> None:
        text = class_source(source(FLAT_CFG), "JK04FlatYawEnvCfg")
        self.assertEqual(assigned_number(text, "self.rewards.yaw_feet_air_time_positive.weight"), 0.0)
        self.assertEqual(assigned_number(text, "self.rewards.feet_gait.weight"), 0.0)
        self.assertEqual(assigned_number(text, "self.rewards.feet_air_time_variance.weight"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
