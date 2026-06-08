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
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground

# 简单低台阶，更适合先检查轮足移动
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene low-stairs

# 明确指定仓库里的 stairs.world 场景
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world

# 只站立打开模型
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --stand

# 进入场景后自动慢速前进
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --speed 0.2

# 固定看整个楼梯场景，不跟随机器人
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --camera fixed

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
