# JK03 本地 MuJoCo 手动仿真

这个入口用于“不训练、不加载策略模型”地在本机打开 JK03 轮足机器人的 3D MuJoCo 仿真。它读取现有 `jk03` URDF，并使用 `robot_lab` 中 JK03 的关节顺序、初始高度、默认关节角、PD 增益、轮关节阻尼和力矩限制。

脚本只会生成临时 MuJoCo URDF，不会修改 `jk03` 源 URDF 或任何 `jk03` 配置参数。

## 推荐启动方式

在仓库根目录执行：

```bash
python3 -m pip install mujoco
python3 rl_sar-main/scripts/manual_mujoco_jk03.py
```

默认会读取：

```text
robot_lab-main/source/robot_lab/data/Robots/jk03/jk03_description/urdf/jk03.urdf
```

并自动修正 mesh 路径、删除临时副本里的 ROS transmission 块、添加 floating base，然后打开 MuJoCo 3D viewer。

默认还会读取仓库里的 `rl_sar-main/src/rl_sar/worlds/stairs.world`，把其中的 ground、静态方块和楼梯 collision box 转成 MuJoCo 临时场景，让 JK03 站在楼梯场景前方，而不是在空场景里漂浮或下落。

## 键盘控制

```text
1      开始轮驱
0      站立
W/S    前进/后退轮速指令
A/D    左右姿态和轮速偏置
Q/E    偏航偏置
Space  清空指令并站立
R      重置姿态
H      显示帮助
Esc    退出 viewer
```

## 常用参数

```bash
# 默认读取 stairs.world，启动后站立不动
python3 rl_sar-main/scripts/manual_mujoco_jk03.py

# 平地
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground --collision-only --camera fixed

# 简单低台阶，更适合先检查轮足移动
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene low-stairs --collision-only --camera fixed

# 明确指定仓库里的 stairs.world 场景
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --collision-only --camera fixed

# 只站立打开模型
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground --collision-only --camera fixed --stand

# 进入场景后自动慢速前进
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene low-stairs --collision-only --camera fixed --speed 0.1

# 固定看整个楼梯场景，不跟随机器人
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --collision-only --camera fixed

# 不开 viewer，跑 5 秒用于检查 MuJoCo 是否能加载模型
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --headless --duration 5

# 关闭软平衡辅助，查看更接近裸物理的效果
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --no-assist

# 如果 MuJoCo 不能加载视觉 mesh，则只显示碰撞几何
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --collision-only
```

可选场景：

```text
empty         不添加地形，保留空场景
ground        平地
low-stairs    低台阶
stairs-world  读取仓库里的 Gazebo stairs.world，默认值
```

默认启用了软平衡辅助，避免未训练控制器马上翻倒。它会根据台阶高度轻微调整目标机身高度，但它不是训练出来的策略，也不适合实机；高楼梯能否稳定通过仍然取决于后续策略和控制效果。狗之前会一直动，是因为旧版本默认给了前进速度；现在默认速度是 0，只有按 `1`、`W/S`，或显式传 `--speed` 才会移动。

## 常见异常和原因

如果看到机器人不停转、漂移、乱飘，通常不是 MuJoCo 坏了，而是因为当前入口只是手写检查控制器，不是训练好的强化学习策略。它只用默认站姿、简单 PD、轮速指令和软平衡辅助来做本地检查；它还没有学会起立、稳定走路、过楼梯或跌倒恢复。

如果看到模型或背景很难看，这是因为 JK03 的部分 STL 视觉 mesh 在 MuJoCo 中可能加载失败，脚本会回退到 `--collision-only` 的简化碰撞几何。这个模式适合检查关节、接触和场景，不代表最终展示效果。

如果看到场景像“消失”，优先判断是相机视角问题，而不是场景被删除。推荐使用固定相机：

```bash
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --collision-only --camera fixed
```

如果按 `0` 或 `1` 后画面显示异常，可能是 MuJoCo viewer 自己也把数字键当作显示组快捷键。此时可以先用 `W/S` 控制速度，或者重新运行固定相机命令。当前脚本中的 `1` 表示开始轮驱，`0` 表示停止站立。

建议按下面顺序检查，不要一开始就上高楼梯：

```bash
# 1. 平地站立，最稳
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground --collision-only --camera fixed --stand

# 2. 平地慢速移动
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground --collision-only --camera fixed --speed 0.1

# 3. 低台阶慢速移动
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene low-stairs --collision-only --camera fixed --speed 0.1

# 4. 仓库 stairs.world 场景，只建议用于场景加载和后续策略验证
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --collision-only --camera fixed
```

真正想让 JK03 稳定走、转向、过楼梯，需要在云服务器上训练 policy，然后再把训练好的 `.pt` 或 `.onnx` 策略拿回 MuJoCo 做验证。这个手动脚本只是模型、场景和接触检查入口。
