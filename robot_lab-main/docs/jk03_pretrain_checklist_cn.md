# JK03 深度学习训练前准备清单

这份清单只做训练前准备，不修改 `jk03` 的初始数据、URDF、网格文件或 robot_lab 里的 JK03 参数。受保护范围包括：

```text
robot_lab-main/source/robot_lab/robot_lab/assets/jk03.py
robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/
robot_lab-main/source/robot_lab/data/Robots/jk03/
```

## 1. 本地预检

在仓库根目录运行：

```bash
cd /Users/admin/Desktop/dog_robot
python3 robot_lab-main/scripts/tools/check_jk03_pretrain.py
```

这个脚本会检查：

- JK03 的 Flat/Rough Gym 环境是否已经注册。
- `rsl_rl` 和 `cusrl` 的 PPO 配置是否存在。
- URDF 里的 16 个关节和网格引用是否能在磁盘上找到。
- `rl_sar-main/scripts/manual_mujoco_jk03.py` 里的关节顺序、初始姿态、PD 参数、力矩/速度限制、动作尺度是否和 JK03 源配置一致。
- git 中受保护的 JK03 源数据路径是否有未提交或已暂存改动。
- 如果本机安装了 `mujoco`，会跑一次 headless 楼梯场景烟测。

如果只想做静态检查，不跑 MuJoCo：

```bash
python3 robot_lab-main/scripts/tools/check_jk03_pretrain.py --skip-mujoco
```

## 2. 已确认的训练入口

JK03 现有环境 ID：

```text
RobotLab-Isaac-Velocity-Flat-JK03-v0
RobotLab-Isaac-Velocity-Rough-JK03-v0
```

建议训练顺序：

```text
Flat-JK03 先训练平地基础速度跟踪
Rough-JK03 再训练粗糙地形和台阶适应
训练后用 play.py 验证，再导出策略
```

## 3. 云服务器训练环境

云服务器建议使用 Linux + NVIDIA GPU。以本仓库 README 为准，当前项目标注的主要环境是 Python 3.11、Isaac Lab、Isaac Sim。云端准备步骤：

```bash
git clone <你的 GitHub 仓库地址> dog_robot
cd dog_robot/robot_lab-main

# 使用已经安装 Isaac Lab / Isaac Sim 的 Python
python -m pip install -e source/robot_lab

# 确认 JK03 环境能被 Isaac Lab 注册
python scripts/tools/list_envs.py --keyword JK03
```

如果 `list_envs.py` 能看到两个 JK03 环境，再做零动作或随机动作小测试：

```bash
python scripts/tools/zero_agent.py --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 --headless --num_envs 16
python scripts/tools/random_agent.py --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 --headless --num_envs 16
```

## 4. 开始训练

先训练平地：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 1024
```

平地策略稳定后，再训练 Rough：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --headless \
  --num_envs 1024
```

如果显存不够，把 `--num_envs` 降到 `512`、`256` 或更低。日志和 checkpoint 会写到：

```text
robot_lab-main/logs/rsl_rl/jk03_flat/
robot_lab-main/logs/rsl_rl/jk03_rough/
```

## 5. 训练后验证和导出

验证平地策略：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --num_envs 16
```

验证 Rough 策略：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --num_envs 16
```

`play.py` 会在 checkpoint 目录下导出 JIT 和 ONNX 策略，通常在：

```text
logs/rsl_rl/<experiment>/<run>/exported/policy.pt
logs/rsl_rl/<experiment>/<run>/exported/policy.onnx
```

## 6. 不要改动的内容

训练前后都不要直接修改 JK03 初始数据。如果需要改奖励、课程、动作尺度或初始姿态，先新建一个派生配置文件或新任务 ID，不要直接覆盖当前 JK03 源配置。每次训练前建议运行：

```bash
python3 robot_lab-main/scripts/tools/check_jk03_pretrain.py --skip-mujoco
```

如果它报告 `protected JK03 data` 失败，先停止训练，检查是不是误改了 `jk03` 源数据。
