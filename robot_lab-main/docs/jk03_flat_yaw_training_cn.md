# JK03 Flat-Yaw 专项训练说明

## Flat-Yaw 是干嘛的

`RobotLab-Isaac-Velocity-Flat-Yaw-JK03-v0` 是 JK03 的平地原地转向专项任务。

普通 `Flat` 同时训练：

- 前进 / 后退：`vx`
- 左右平移：`vy`
- 左右转向：`yaw`

所以普通 `Flat` 里，原地 yaw 转向只占一部分样本，抬腿/步态信号会被前进和横移任务稀释。

`Flat-Yaw` 则把命令范围收窄为：

- `vx = 0`
- `vy = 0`
- 只训练 `ang_vel_z`

它的目的不是直接训练完整平地运动，而是先让策略集中学会：

- 按 `Z/X` 时能产生 yaw 转向响应。
- 原地转向时不要只靠后轮拖滑。
- 原地转向时开始形成更像步态的接触节奏。
- 训练出 yaw 能力后，再迁移/参考到普通 `Flat`。

## 云端训练命令

```bash
cd /root/dog-robot-main/robot_lab-main
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK03-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 10000
```

## 日志位置

Flat-Yaw 的 TensorBoard 日志会写到：

```bash
logs/rsl_rl/jk03_flat_yaw/
```

不要和普通 Flat 混淆：

```bash
logs/rsl_rl/jk03_flat/
```

## 测试 checkpoint

测试时 task 也必须用 `Flat-Yaw`，否则环境配置和训练配置不一致。

把下面的 `你的run目录` 和 `model_1000.pt` 换成实际文件名：

```bash
cd /root/dog-robot-main/robot_lab-main
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab

/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK03-v0 \
  --checkpoint logs/rsl_rl/jk03_flat_yaw/你的run目录/model_1000.pt \
  --keyboard \
  --num_envs 1 \
  --real-time
```

## 键盘控制

在 `--keyboard` 模式下：

- `Z`：正方向 yaw
- `X`：负方向 yaw
- 方向键 / WASD：平移命令

但在 Flat-Yaw 中，训练命令本身是 `vx=0, vy=0`，所以这个任务重点是看 `Z/X` 的原地转向能力。

## 判断是否有效

不要只看单点，建议按 200-step 窗口均值看：

- `Episode_Reward/track_ang_vel_z_exp` 是否上升。
- `Metrics/base_velocity/error_vel_yaw` 是否下降。
- `Episode_Reward/feet_air_time` 是否从负值逐渐接近 0 或转正。
- `Episode_Reward/feet_gait` 是否上升。
- `Episode_Reward/feet_slide` 是否没有继续恶化。
- 视频里是否仍然只是 hipx 拧腿 / 后轮拖滑。

如果 Flat-Yaw 到 1000-1500 步仍完全不抬腿，说明只靠 `feet_air_time + feet_gait` 还不够，需要考虑更明确的 contact schedule、reference motion 或 imitation-style 先验。
