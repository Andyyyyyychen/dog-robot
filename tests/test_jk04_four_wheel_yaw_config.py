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
        self.assertIn("front_left_score", text)
        self.assertIn("front_right_score", text)

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

    def test_flat_yaw_penalizes_rear_only_front_wheel_underuse(self) -> None:
        rough_text = source(ROUGH_CFG)
        tree = ast.parse(rough_text)
        function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn("yaw_rear_only_front_wheel_penalty", function_names)
        self.assertIn("yaw_rear_only_front_wheel_penalty = RewTerm(", rough_text)
        self.assertIn("func=yaw_rear_only_front_wheel_penalty", rough_text)
        self.assertIn('"front_min_wheel_speed"', rough_text)
        self.assertIn('"rear_target_wheel_speed"', rough_text)

        flat_yaw_text = class_source(source(FLAT_CFG), "JK04FlatYawEnvCfg")
        self.assertLess(
            assigned_number(flat_yaw_text, "self.rewards.yaw_rear_only_front_wheel_penalty.weight"),
            0.0,
        )
        self.assertIn(
            'self.rewards.yaw_rear_only_front_wheel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names',
            flat_yaw_text,
        )

    def test_flat_yaw_rebalances_wheel_rewards_for_stability(self) -> None:
        text = class_source(source(FLAT_CFG), "JK04FlatYawEnvCfg")
        self.assertLessEqual(
            assigned_number(text, "self.rewards.yaw_front_rear_wheel_participation.weight"),
            1.10,
        )
        self.assertGreater(
            assigned_number(text, "self.rewards.yaw_rear_only_front_wheel_penalty.weight"),
            -1.25,
        )
        self.assertLessEqual(assigned_number(text, "self.rewards.lin_vel_z_l2.weight"), -0.85)
        self.assertLessEqual(assigned_number(text, "self.rewards.ang_vel_xy_l2.weight"), -0.20)
        self.assertLessEqual(assigned_number(text, "self.rewards.action_rate_l2.weight"), -0.012)
        self.assertLess(assigned_number(text, "self.rewards.joint_acc_wheel_l2.weight"), 0.0)

    def test_flat_yaw_enables_supported_step_shaping(self) -> None:
        text = class_source(source(FLAT_CFG), "JK04FlatYawEnvCfg")
        self.assertGreater(assigned_number(text, "self.rewards.yaw_feet_air_time_positive.weight"), 0.0)
        self.assertGreater(assigned_number(text, "self.rewards.feet_gait.weight"), 0.0)
        self.assertLess(assigned_number(text, "self.rewards.feet_air_time_variance.weight"), 0.0)
        self.assertGreater(assigned_number(text, "self.rewards.yaw_inside_hind_step_participation.weight"), 0.0)
        self.assertIn(
            'self.rewards.yaw_inside_hind_step_participation.params["left_hind_body_name"] = "hl_wheel"',
            text,
        )
        self.assertIn(
            'self.rewards.yaw_inside_hind_step_participation.params["right_hind_body_name"] = "hr_wheel"',
            text,
        )

    def test_flat_yaw_penalizes_in_place_xy_drift(self) -> None:
        text = class_source(source(FLAT_CFG), "JK04FlatYawEnvCfg")
        self.assertLess(assigned_number(text, "self.rewards.yaw_in_place_xy_drift_penalty.weight"), 0.0)
        self.assertIn(
            'self.rewards.yaw_in_place_xy_drift_penalty.params["velocity_threshold"] = 0.30',
            text,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
