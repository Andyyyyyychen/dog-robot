# Go2 本地 MuJoCo 手动仿真

这个入口用于“不训练、不加载策略模型”地在本机打开 Go2 机器狗的 3D MuJoCo 仿真。它使用程序化步态和 PD 力矩控制，方便你先看模型姿态、关节响应和简单运动；后续真正训练仍然建议放到云服务器/Isaac Lab。

## 推荐启动方式

在仓库根目录执行：

```bash
python3 -m pip install mujoco
python3 rl_sar-main/scripts/manual_mujoco_go2.py
```

默认会读取：

```text
robot_lab-main/source/robot_lab/data/Robots/unitree/go2_description/urdf/go2_description.urdf
```

并自动修正 mesh 路径、生成临时 MuJoCo URDF，然后打开 3D viewer。

## 键盘控制

```text
1      开始程序化小跑
0      站立
W/S    前进/后退速度
A/D    左右偏置
Q/E    偏航偏置
Space  清空速度并站立
R      重置姿态
H      显示帮助
Esc    退出 viewer
```

## 常用参数

```bash
# 只站立打开模型
python3 rl_sar-main/scripts/manual_mujoco_go2.py --stand

# 不开 viewer，跑 5 秒用于检查 MuJoCo 是否能加载模型
python3 rl_sar-main/scripts/manual_mujoco_go2.py --headless --duration 5

# 关闭软平衡辅助，查看更接近裸物理的效果
python3 rl_sar-main/scripts/manual_mujoco_go2.py --no-assist

# 如果你的 MuJoCo 不能加载 DAE 视觉 mesh，则只显示碰撞几何
python3 rl_sar-main/scripts/manual_mujoco_go2.py --collision-only
```

默认启用了软平衡辅助，避免未训练步态马上翻倒。它不是训练出来的策略，也不适合实机，只是本地调模型/看动作的辅助控制。脚本会先尝试加载完整视觉 mesh；如果 MuJoCo 对 `.dae` mesh 不兼容，会自动退回到 collision-only 简化显示。

## 原 C++ MuJoCo 入口

我也给原来的 `rl_sim_mujoco` 加了 `--manual` 参数。等你补齐 `rl_sar_zoo` 的 MJCF 场景并完成 MuJoCo 编译后，可以这样运行：

```bash
cd rl_sar-main
./build.sh -mj
./cmake_build/bin/rl_sim_mujoco go2 scene --manual
```

如果当前工程里没有 `rl_sar_zoo/go2_description/mjcf/scene.xml`，优先使用上面的 Python 入口，它直接复用本仓库已有的 `robot_lab-main` Go2 URDF。
