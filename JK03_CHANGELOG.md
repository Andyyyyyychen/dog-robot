# JK03 Change Log

本文件记录 JK03 训练代码、奖励函数、PPO 配置和测试流程的关键修改。以后每次改代码都必须同步更新本文件，并在确认无误后提交到 GitHub。

## 固定规则

每次修改都要记录：

- 修改时间和版本名。
- 为什么修改：对应的问题、指标或测试现象。
- 修改文件：列出具体路径。
- 怎么修改：函数、参数、权重、逻辑变化要写清楚。
- 没有修改什么：尤其说明是否保持 URDF、`jk03.py`、terrain curriculum 不变。
- 验证结果：至少写本地编译、云端编译、云端上传状态。
- 已知风险：训练后可能出现的问题。
- 下一步观察指标：例如 `terrain_levels`、`commanded_motion_progress`、`stair_upward_progress`、`wheel_clearance_on_command`、`feet_slide`、`wheel_lateral_edge_contact`。

受保护内容：

- 不修改 `robot_lab-main/source/robot_lab/robot_lab/assets/jk03.py` 中的 JK03 初始物理数据。
- 不修改 `robot_lab-main/source/robot_lab/data/Robots/jk03/jk03_description/urdf/jk03.urdf`。
- 不修改 fan-ziqi 原始 terrain curriculum 算法：`terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
- 不在 JK03 rough 配置里覆盖 terrain level 算法。

可以修改的内容：

- JK03 reward 函数和 reward 权重。
- PPO 训练超参数。
- play/debug 工具。
- README 和训练说明文档。

## 2026-06-16: flat-yaw-air-clearance-v1

状态：本地验证通过，已同步云端，等待 GitHub 提交。

### 为什么修改

Flat 新版训练到约 1600 step 后，用户实测仍然没有明显抬腿/抬轮转向。TensorBoard 数据也支持这个判断：

- `vel_yaw_error` 在下降，说明它越来越会产生 yaw。
- `joint_deviation_hipx_l1` 和 `yaw_turn_joint_posture_l2` 在改善，说明强拧 hipx 减少了。
- 但 `feet_gait` 均值没有稳定突破，`feet_slide` 仍然略差，说明策略可能仍在贴地滑动/拖地转。

因此只靠 `feet_gait` 的接触时序奖励不够，需要给它一个更直接、但严格受限的“原地 yaw 抬轮/抬脚 clearance”奖励。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 新增 `yaw_turn_feet_clearance`

新增 reward 函数：

```python
yaw_turn_feet_clearance
```

这个函数只在以下条件同时满足时给奖励：

```text
abs(yaw_command) > command_threshold
xy_command_norm <= max_xy_command
current_air_time > min_air_time
base_height >= min_base_height
robot upright
```

也就是说：

- 必须是接近原地左/右转。
- 轮/脚必须真的离地，不能只靠身体趴低骗 clearance。
- 身体高度太低时不给分。
- 机器人歪倒时不给分。

核心计算：

```python
clearance = clamp((foot_z_body - min_height) / (target_height - min_height), 0, 1)
swing_scale = tanh(tanh_mult * horizontal_foot_speed)
airborne = current_air_time > min_air_time
short_air = clamp((max_air_time - current_air_time) / (max_air_time - min_air_time), 0, 1)
reward = mean(clearance * swing_scale * airborne * short_air)
```

含义：

- `clearance`：轮/脚相对机身抬得够不够。
- `swing_scale`：轮/脚是否真的在摆动换位置。
- `airborne`：必须离地。
- `short_air`：鼓励短促抬步，不鼓励长时间悬空乱跳。

#### JK03 reward 配置里注册新 term

在 `JK03RewardsCfg` 中新增：

```python
yaw_turn_feet_clearance = RewTerm(...)
```

默认权重为：

```python
weight = 0.0
```

这样 Rough 不会自动受影响，只有 Flat 显式打开时才生效。

参数：

```python
min_height = -0.35
target_height = -0.29
min_air_time = 0.015
max_air_time = 0.20
min_base_height = 0.425
base_height_margin = 0.025
```

这些值的含义：

- 默认轮心相对 base 大约在 `-0.35m` 附近。
- `target_height=-0.29` 表示希望转向时轮/脚能相对机身抬起约 6cm。
- `min_air_time=0.015` 防止刚刚抖一下就算有效抬腿。
- `max_air_time=0.20` 防止它长时间悬空乱跳。
- `min_base_height=0.425` 防止趴下骗奖励。

#### Flat 中打开 yaw-only 抬轮奖励

新增：

```python
self.rewards.yaw_turn_feet_clearance.weight = 0.80
self.rewards.yaw_turn_feet_clearance.params["command_threshold"] = 0.06
self.rewards.yaw_turn_feet_clearance.params["max_xy_command"] = 0.18
self.rewards.yaw_turn_feet_clearance.params["asset_cfg"].body_names = [self.foot_link_name]
self.rewards.yaw_turn_feet_clearance.params["sensor_cfg"].body_names = [self.foot_link_name]
```

目的：

- 只在原地转向时奖励抬轮/抬脚。
- 使用 JK03 的四个 wheel body 作为 foot/wheel。
- 结合已有 `feet_gait`，让它不只是接触节奏对，还要真的离地。

#### 微调 Flat 动作空间

修改：

```python
hipy/knee action scale: 0.14 -> 0.17
```

原因：

- 之前 hipy/knee 动作幅度可能太小，策略即使想抬腿也难以产生足够 clearance。
- `hipx` 仍保持 `0.06`，继续限制大幅横向拧腿。

#### 加强滑动惩罚

修改：

```python
feet_slide.weight: -0.12 -> -0.16
```

原因：

- 既然现在给了抬轮/抬脚奖励，就可以稍微提高滑动惩罚，减少贴地拖着转的投机方式。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改 Rough 楼梯 reward 权重。
- 没改 PPO。

### 验证

- 本地 Python 编译检查通过：
  - `rewards.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
- 本地 `git diff --check` 通过。
- 本地保护项 diff 为空：
  - 未修改 `jk03.py`。
  - 未修改 `jk03.urdf`。
  - 未修改 `velocity_env_cfg.py`。
  - 未修改 `curriculums.py`。
- 云端 `ssh -p 30216 root@183.147.142.40` 已覆盖上传。
- 云端 `python3 -m py_compile` 通过。
- 云端确认 `velocity_env_cfg.py` 仍为 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
- 云端确认 JK03 rough 没有 `terrain_levels` 覆盖：`NO_JK03_TERRAIN_OVERRIDE`。
- GitHub commit/push：待本次版本提交并推送。

### 已知风险

- 如果 `yaw_turn_feet_clearance` 太强，可能出现转向时小跳步，需要降低 weight 或降低 target clearance。
- 如果 `feet_slide` 惩罚太强，可能短期转向速度下降。
- 如果 2000-2500 step 后仍不抬腿，需要检查 contact sensor 的 `current_air_time` 是否真的能捕捉 wheel 离地。

### 下一步观察指标

- `Episode_Reward/yaw_turn_feet_clearance`
- `Episode_Reward/feet_gait`
- `Episode_Reward/feet_slide`
- `Metrics/base_velocity/error_vel_yaw`
- `Episode_Reward/ang_vel_xy_l2`
- 视频里 wheel/foot 是否短促离地，而不是贴地滑动。

## 2026-06-16: flat-yaw-diagonal-gait-v1

状态：本地验证通过，已同步云端，已提交并推送 GitHub。

### 为什么修改

用户和同事反馈 Flat 原地转向仍然不对：

- 转向时不应该靠强拧 hipx/腿部角度完成。
- 成熟四足原地转向通常是左右/对角腿有节奏地踏步换向。
- 当前 Flat 即使 yaw 指标在变好，视频里的动作仍可能是“拧关节、拖地、趴低”。

本次目标是只做精炼修改：不再新增复杂抬腿函数，而是复用仓库已有的 `GaitReward`，让它只在原地 yaw 命令下启用，给策略明确的“对角腿同步、另一对对角腿反相”的踏步转向信号。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 扩展 `GaitReward`

在 `GaitReward.__call__` 增加两个可选参数：

```python
yaw_command_only: bool = False
max_xy_command: float | None = None
```

含义：

- 默认 `False`，所以其他机器人和 Rough 原来的 gait reward 行为不变。
- 当 `yaw_command_only=True` 时，只有满足：

```text
abs(yaw_command) > command_threshold
xy_command_norm <= max_xy_command
```

才启用 gait reward。

这样 Flat 原地转向会被鼓励成“对角踏步转”，但直走/后退时不会被这个转向步态奖励干扰。

#### Flat 打开 yaw-only 对角步态奖励

新增/调整：

```python
feet_gait.weight = 0.35
feet_gait.command_threshold = 0.06
feet_gait.velocity_threshold = 0.10
feet_gait.max_err = 0.25
feet_gait.synced_feet_pair_names = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
feet_gait.yaw_command_only = True
feet_gait.max_xy_command = 0.18
```

目的：

- `fl_wheel + hr_wheel` 同步。
- `fr_wheel + hl_wheel` 同步。
- 两组对角腿互相反相。
- 只在近似原地左/右转命令下生效。

这比之前单独奖励“脚抬高”更稳，因为它奖励的是接触时序，而不是单纯让脚离地。

#### 适度放松 yaw 时 hipx 硬约束

修改：

```python
yaw_turn_joint_posture_l2.weight: -1.10 -> -0.75
```

原因：

- `-1.10` 会强压 hipx 偏离，确实能减少拧腿，但也可能让策略不敢正常小幅摆腿换步。
- 改到 `-0.75` 后仍然惩罚大幅拧 hipx，但允许必要的小幅踏步调整。

#### 适度增强 yaw 目标，但不回到过强版本

修改：

```python
track_ang_vel_z_exp.weight: 4.0 -> 4.5
yaw_command_progress.weight: 2.0 -> 2.4
yaw_command_progress.max_yaw_rate: 0.60 -> 0.70
ang_vel_z range: (-0.8, 0.8) -> (-0.9, 0.9)
```

原因：

- 右转/原地 360 度仍弱，yaw 目标需要稍强一点。
- 但不恢复到上一版 `7.0 / 4.0 / 1.2` 那种强 yaw，因为那会诱导趴下、侧倾、扭腿硬转。

#### 放松一点 feet_slide

修改：

```python
feet_slide.weight: -0.18 -> -0.12
```

原因：

- 原地转向时脚/轮需要换支撑点，过强滑动惩罚可能让策略宁愿卡住不转。
- 这不是鼓励拖地，而是给 gait reward 留出踏步探索空间。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改 Rough 楼梯 reward 权重。
- 没改 PPO。

### 验证

- 本地 Python 编译检查通过：
  - `rewards.py`
  - `flat_env_cfg.py`
  - `rough_env_cfg.py`
- 本地 `git diff --check` 通过。
- 本地保护项 diff 为空：
  - 未修改 `jk03.py`。
  - 未修改 `jk03.urdf`。
  - 未修改 `velocity_env_cfg.py`。
  - 未修改 `curriculums.py`。
- 云端 `ssh -p 30216 root@183.147.142.40` 已覆盖上传。
- 云端 `python3 -m py_compile` 通过。
- 云端确认 `velocity_env_cfg.py` 仍为 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
- 云端确认 JK03 rough 没有 `terrain_levels` 覆盖：`NO_JK03_TERRAIN_OVERRIDE`。
- GitHub commit/push：`1ade8dc Add JK03 flat yaw diagonal gait reward`。

### 已知风险

- gait reward 是接触时序奖励，不是显式动作相位控制；它会引导踏步，但不能保证一开始就像实机控制器那样稳定。
- 如果 `feet_gait` 太强，可能让原地转向出现小跳步；如果太弱，则仍可能靠轮子拖地。
- 如果右转继续比左转弱，需要单独做 yaw 正负方向对称性测试。

### 下一步观察指标

- `Episode_Reward/feet_gait`
- `Metrics/base_velocity/error_vel_yaw`
- `Episode_Reward/yaw_command_progress`
- `Episode_Reward/yaw_turn_joint_posture_l2`
- `Episode_Reward/feet_slide`
- 视频里左右转是否从“拧 hipx”变成“对角短步换向”。

## 2026-06-16: flat-turn-simplify-after-video-v1

状态：本地验证通过，已同步云端，准备提交 GitHub。

### 为什么修改

用户反馈上一版 Flat 仍然有严重问题：

- 左右转变成趴下身体左右转。
- 后退很慢。
- 转弯仍然靠扭动关节，不是抬腿/换步。
- 上一版新增函数太多，reward 目标过散。

我从云端下载并重新录制了最新 Flat 视频：

- 云端视频：`logs/rsl_rl/jk03_flat/2026-06-16_11-03-29/videos/play/rl-video-step-0.mp4`
- 本地视频：`cloud_videos/flat_latest_2026-06-16_110329.mp4`
- 本地抽帧：`cloud_videos/flat_latest_2026-06-16_110329_frames/`

视频观察结论：

- 机器人确实不是原地踏步转向。
- 转向时机身明显侧倾/压低。
- 轮子外撇，腿部 hipx/hipy/knee 姿态很大。
- 轮子多数时间接近贴地拖动，不是稳定小步换向。
- 上一版 `yaw_turn_feet_air_time` 和 `yaw_turn_feet_clearance` 没有让它学会真正踏步，反而和过强 yaw reward 叠加，给了“趴下、撑开、拖地转”的投机空间。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 删除上一版多余 yaw-only 函数

删除：

```python
yaw_turn_feet_air_time
yaw_turn_feet_clearance
```

原因：

- 它们没有可靠形成“像成熟四足那样小步换向”的动作。
- 对轮腿 JK03 来说，仅奖励轮/脚短暂离地或相对高度，容易被策略用侧倾、外撇、拖地方式骗过去。
- 本次改为更精炼的 Flat 目标，不继续堆函数。

#### 删除对应 reward term

从 JK03 reward 配置里删除：

```python
yaw_turn_feet_air_time
yaw_turn_feet_clearance
```

保留：

```python
front_joint_posture_l2
yaw_turn_joint_posture_l2
```

它们都复用现有 `commanded_joint_posture_l2`，不新增复杂函数。

#### Flat 中关闭 rough/stair 类 reward

Flat 中关闭：

```python
commanded_motion_progress.weight = 0
upward_without_forward_progress.weight = 0
vertical_bounce_without_progress.weight = 0
wheel_spin_without_progress.weight = 0
wheel_clearance_on_command.weight = 0
feet_gait.weight = 0
upward.weight = 0
```

原因：

- Flat 的目标是基础直走、后退、左右原地转。
- 楼梯/抬升/越障类 reward 会给平地策略奇怪的投机方向。

#### Flat 中加强机身水平和站高

新增/调整：

```python
flat_orientation_l2.weight = -2.0
ang_vel_xy_l2.weight = -0.15
commanded_base_height_below_target.weight = -3.5
```

原因：

- 视频里主要问题不是转不动，而是转向时身体侧倾、压低。
- 先保证机身水平和站高，再谈丝滑转向。

#### Flat 中限制腿部 action 幅度

新增：

```python
actions.joint_pos.scale = {
    ".*_hipx_joint": 0.06,
    ".*_hipy_joint": 0.14,
    ".*_knee_joint": 0.14,
}
actions.joint_vel.scale = 10.0
```

原因：

- 平地转向应该更多靠轮速/差速和小姿态调整，不应该靠大幅扭腿。
- 降低腿部 action 幅度，给轮子动作更多空间。

#### Flat 中降低过强 yaw 奖励

从上一版：

```python
track_ang_vel_z_exp.weight = 7.0
yaw_command_progress.weight = 4.0
yaw_stuck_with_command.weight = -5.0
ang_vel_z = (-1.2, 1.2)
```

调整为：

```python
track_ang_vel_z_exp.weight = 4.0
yaw_command_progress.weight = 2.0
yaw_stuck_with_command.weight = -3.0
ang_vel_z = (-0.8, 0.8)
```

原因：

- 上一版 yaw 目标太强，策略宁愿趴下、侧倾、扭腿也要追 yaw。
- 现在先让 yaw 能稳定、水平地转，再逐步提高速度。

#### Flat 中改善后退慢

新增：

```python
commands.base_velocity.ranges.lin_vel_x = (-0.8, 1.0)
```

原因：

- 之前 Flat 继承的后退范围偏小，训练/键盘后退自然慢。
- 这次直接把后退命令范围扩大到 `-0.8`。

#### Flat 中继续抑制前腿前折和 hipx 扭转

调整：

```python
front_joint_posture_l2.weight = -0.70
yaw_turn_joint_posture_l2.weight = -1.10
joint_deviation_hipx_l1.weight = -0.35
joint_pos_penalty.weight = -1.1
feet_slide.weight = -0.18
```

原因：

- 前腿前折继续用前腿 hipy/knee 姿态约束处理。
- 转向扭腿主要压 `hipx`，避免大幅外撇。
- 增加 `feet_slide` 惩罚，减少拖地横滑。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改 command generator 通用逻辑。
- 本次主要修 Flat，不动 Rough 楼梯 curriculum。

### 验证

- 已下载/录制最新 Flat 视频并查看抽帧。
- 本地 Python 编译检查通过：
  - `rewards.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
- 本地 `git diff --check` 通过。
- 本地保护项 diff 为空：
  - 未修改 `jk03.py`。
  - 未修改 `jk03.urdf`。
  - 未修改 `velocity_env_cfg.py`。
  - 未修改 `curriculums.py`。
- 云端 `ssh -p 30216 root@183.147.142.40` 已覆盖上传。
- 云端 `python3 -m py_compile` 通过。
- 云端确认 `velocity_env_cfg.py` 仍为 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
- 云端确认 JK03 rough 没有 `terrain_levels` 覆盖：`NO_JK03_TERRAIN_OVERRIDE`。
- GitHub commit/push：随本次版本提交并推送。

### 已知风险

- yaw reward 降低后，早期原地转速度可能变慢，但应该更少趴下和扭腿。
- 腿部 action scale 降低后，Flat 会更偏轮式运动；如果用户坚持必须像纯四足那样明显抬腿，需要后续单独设计 gait phase 或接触时序，而不是再堆普通 reward。
- 如果右转仍弱，下一步必须做固定 `yaw=-0.6` 和 `yaw=+0.6` 的对称测试，检查符号和轮速差是否对称。

### 下一步观察指标

- 视频中 base 是否保持水平。
- 转向时是否还明显趴下。
- `Episode_Reward/yaw_turn_joint_posture_l2`
- `Episode_Reward/front_joint_posture_l2`
- `Episode_Reward/feet_slide`
- `Metrics/base_velocity/error_vel_yaw`
- 后退命令下 `base_vx` 是否能明显接近负值。

## 2026-06-16: flat-yaw-step-turn-v1

状态：本地验证通过，已同步云端，准备提交 GitHub。

### 为什么修改

用户测试最新版 Flat 后反馈：

- 直走已经明显变好，但前轮/前腿仍然有轻微向前弯曲。
- 左右原地转向还不能稳定完成 360 度。
- 右转尤其弱。
- 当前转弯方式更像把关节拧过去，不像成熟四足那样靠抬腿/换步完成原地转向。

本次目标是把 Flat 的转向策略从“扭关节硬转”引导到“yaw 命令下短步抬轮/抬腿换向”。参考了仓库内 Unitree Go2、A1、MagicDog 等成熟四足配置常见思路：yaw tracking、feet air time、feet height/clearance、joint mirror/姿态约束组合使用，而不是单靠 yaw 速度奖励。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 扩展 `commanded_joint_posture_l2`

新增两个可选参数：

```python
yaw_command_only: bool = False
max_xy_command: float | None = None
```

作用：

- 原来只能按“有命令”或“直走命令”启用。
- 现在可以按“近似原地 yaw 命令”启用。
- 用于单独限制转向时的 `hipx` 侧摆，避免靠大幅拧腿完成转向。

#### 新增 `yaw_turn_feet_air_time`

新增 yaw-only 抬步奖励：

```python
yaw_turn_feet_air_time
```

逻辑：

- 只有 yaw 命令超过阈值，并且 xy 命令很小时才启用。
- 当轮/脚离地一小段时间后重新接触地面，会得到奖励。
- 奖励窗口是 `0.04s` 到 `0.22s`，鼓励短促换步，而不是长时间悬空。

目的：

- 让原地转向时学会“抬一下再落下”的换步，而不是一直贴地拧。

#### 新增 `yaw_turn_feet_clearance`

新增 yaw-only 抬轮/抬脚高度奖励：

```python
yaw_turn_feet_clearance
```

逻辑：

- 只在近似原地 yaw 命令下启用。
- 计算 wheel/foot 相对 base 的 z 高度。
- 鼓励移动中的轮/脚从约 `-0.34m` 抬到约 `-0.28m` 附近。

目的：

- 给策略一个连续信号，告诉它转向时不要只拖地或拧腿，要把轮/脚稍微抬起来换位置。

#### Flat 中调整直走姿态

```python
commanded_joint_posture_l2.straight_command_only = True
commanded_joint_posture_l2.max_abs_yaw_command = 0.08
```

含义：

- 原来 Flat 的关节姿态约束对 yaw 也生效。
- 现在它只管直走/后退时保持默认姿态。
- 转向时不再用全腿默认姿态约束锁死 hipy/knee，给抬腿换步留空间。

新增前腿直走约束：

```python
front_joint_posture_l2.weight = -0.55
front_joint_posture_l2.joint_names = [
    "fl_hipy_joint", "fl_knee_joint", "fr_hipy_joint", "fr_knee_joint",
]
```

目的：

- 直走时专门压住前腿 hipy/knee 的过度前折。
- 只在直走命令下生效，不限制原地转向抬腿。

#### Flat 中调整原地转向

新增 yaw-only hipx 姿态约束：

```python
yaw_turn_joint_posture_l2.weight = -0.55
yaw_turn_joint_posture_l2.joint_names = ".*_hipx_joint"
yaw_turn_joint_posture_l2.yaw_command_only = True
```

目的：

- 原地转向时不希望靠 hipx 大幅侧摆/拧腿完成动作。

打开 yaw-only 抬步奖励：

```python
yaw_turn_feet_air_time.weight = 0.45
yaw_turn_feet_clearance.weight = 0.55
```

目的：

- 让策略在原地转向时更倾向抬轮/抬脚换步。

加强 yaw 控制：

```python
track_ang_vel_z_exp.weight = 7.0
yaw_command_progress.weight = 4.0
yaw_command_progress.max_yaw_rate = 0.85
yaw_stuck_with_command.weight = -5.0
commands.base_velocity.ranges.ang_vel_z = (-1.2, 1.2)
```

目的：

- 给左右转更大的训练范围和更明确的奖励。
- 让 360 度原地转向更可能学出来。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改 command generator 通用逻辑。
- Rough 中新增 reward term 默认权重都是 `0.0`，本次主要生效对象是 Flat。

### 验证

- 本地 Python 编译检查通过：
  - `rewards.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
- 本地 `git diff --check` 通过。
- 本地保护项 diff 为空：
  - 未修改 `jk03.py`。
  - 未修改 `jk03.urdf`。
  - 未修改 `velocity_env_cfg.py`。
  - 未修改 `curriculums.py`。
- 云端 `ssh -p 30216 root@183.147.142.40` 已覆盖上传。
- 云端 `python3 -m py_compile` 通过。
- 云端确认 `velocity_env_cfg.py` 仍为 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
- 云端确认 JK03 rough 没有 `terrain_levels` 覆盖：`NO_JK03_TERRAIN_OVERRIDE`。
- GitHub commit/push：随本次版本提交并推送。

### 已知风险

- yaw-only 抬步奖励可能让训练早期转向动作更活跃，需要重新训练 Flat 后观察视频。
- 如果出现转向时跳动太大，优先降低 `yaw_turn_feet_clearance.weight`。
- 如果仍然右转弱，下一步要检查键盘命令符号、yaw 正负方向和左右轮/对角腿动作是否存在结构性偏置。
- 如果前腿直走仍前折，可以继续提高 `front_joint_posture_l2.weight`，但不要过高，否则会限制正常缓冲。

### 下一步观察指标

- `Episode_Reward/yaw_turn_feet_air_time`
- `Episode_Reward/yaw_turn_feet_clearance`
- `Episode_Reward/yaw_turn_joint_posture_l2`
- `Episode_Reward/front_joint_posture_l2`
- `Metrics/base_velocity/error_vel_yaw`
- 视频中原地左转/右转是否都能持续旋转。
- 转向时是否从“拧 hipx”变成“短步抬轮/抬腿换向”。

## 2026-06-16: urdf-height-flat-turn-fix-v1

状态：已本地验证，已同步云端，随本次提交推送 GitHub。

### 为什么修改

用户测试 Flat 后发现两个核心问题：

- Flat 策略一跑就明显蹲下去，四肢关节弯曲，重心降低。
- 键盘左右不能实现原地转弯。

这说明上一版只靠 `commanded_base_height_below_target` 还不够，原因有两个：

- 目标高度 `0.43` 偏低，没有真正按 URDF 的轮子半径和默认站姿计算。
- 只有 base 高度惩罚，不足以阻止腿部关节偏离默认姿态。
- Flat 的 yaw 命令范围和 yaw 奖励偏保守，键盘原地转信号不够强。

### URDF 依据

根据 `jk03.urdf`：

- `hipy -> knee` 关节原点 z 长度约 `0.20m`。
- `knee -> wheel` 关节原点 z 长度约 `0.2455886m`。
- wheel collision 半径是 `0.105m`。
- `jk03.py` 默认站姿是：

```python
hipy = 0.9
knee = -1.33
```

按 URDF 前向几何计算，默认站姿下 wheel center 相对 base 的 z 约为：

```text
-0.3509m
```

所以轮子接地时，理论 base 高度约为：

```text
0.3509 + 0.105 = 0.4559m
```

因此这次将训练 reward 的目标高度从 `0.43` 改为 `0.456`。这不是修改 JK03 初始参数，而是让 reward target 与 URDF 几何更一致。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 新增 `commanded_joint_posture_l2`

新增 reward 函数：

```python
commanded_joint_posture_l2
```

逻辑：

```python
joint_error = joint_pos - default_joint_pos
reward = mean(square(joint_error))
```

含义：

- 有命令时，如果腿部关节偏离默认站姿太多，就扣分。
- Flat 中对所有速度/yaw 命令生效。
- Rough 中只对直走命令生效，避免影响转向和爬楼梯姿态调整。

#### Flat 高度和姿态加强

Flat 中修改：

```python
base_height_l2.weight = -1.0
base_height_l2.target_height = 0.456
commanded_base_height_below_target.weight = -3.0
commanded_base_height_below_target.target_height = 0.456
commanded_base_height_below_target.height_margin = 0.06
commanded_joint_posture_l2.weight = -0.8
joint_pos_penalty.weight = -0.9
```

目的：

- 不再允许 Flat 策略通过蹲低获得速度。
- 直走/后退/yaw 时都更倾向保持默认腿部姿态。

#### Flat 原地转向加强

Flat 中修改：

```python
track_ang_vel_z_exp.weight = 6.0
yaw_command_progress.weight = 3.0
yaw_command_progress.max_yaw_rate = 0.60
yaw_stuck_with_command.weight = -4.0
commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
command_levels_ang_vel.range_multiplier = (0.8, 1.0)
```

目的：

- 让训练更重视 yaw。
- 让键盘左右键能给出更明显的 yaw command。
- 让纯 yaw / 原地转向样本更容易被学到。

#### Rough 直走姿态同步修正

Rough 中修改：

```python
base_height_l2.target_height = 0.456
commanded_base_height_below_target.weight = -0.50
commanded_base_height_below_target.target_height = 0.456
commanded_joint_posture_l2.weight = -0.25
commanded_joint_posture_l2.straight_command_only = True
```

目的：

- Rough 直走时也不要明显趴低。
- 但 Rough 不像 Flat 那样强约束，避免让楼梯动作变僵硬。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改 terrain generator。

### 验证

- 本地 Python 编译检查通过：
  - `rewards.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
- 本地 `git diff --check` 通过。
- 本地保护项 diff 为空：
  - 未修改 `jk03.py`。
  - 未修改 `jk03.urdf`。
  - 未修改 `velocity_env_cfg.py` 中的 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
  - 未修改 `curriculums.py`。
- 云端 `ssh -p 30216 root@183.147.142.40` 已覆盖上传。
- 云端 `python3 -m py_compile` 通过。
- 云端确认 `rough_env_cfg.py` 没有 `terrain_levels` 覆盖：`NO_JK03_TERRAIN_OVERRIDE`。
- GitHub commit/push：随本次版本提交并推送。

### 已知风险

- Flat 姿态约束明显变强，早期训练可能速度学得慢一点。
- Rough 直走姿态会更稳，但如果爬楼梯变僵，需要降低 Rough 的 `commanded_base_height_below_target.weight` 或 `commanded_joint_posture_l2.weight`。
- 如果原地转向仍弱，下一步应检查左右轮差速是否学出来，而不是继续只加 yaw reward。

### 下一步观察指标

- Flat：
  - `Episode_Reward/commanded_base_height_below_target`
  - `Episode_Reward/commanded_joint_posture_l2`
  - `Episode_Reward/yaw_command_progress`
  - `Metrics/base_velocity/error_vel_yaw`
  - `Metrics/base_velocity/error_vel_xy`
- 视频/键盘：
  - 直走时 base 是否接近站高。
  - 四肢 hipy/knee 是否还明显继续弯。
  - 只按左右键时能否原地转向。

## 2026-06-16: rough-straight-posture-hold-v1

状态：本地最新版，已准备同步云端和 GitHub。

### 为什么修改

用户进一步说明：不仅希望 Flat 平地动作好，也希望 Rough 训练后的狗在直走时不要明显改变姿态，不要按前进后四肢弯曲、重心降低。

上一版 `commanded_base_height_below_target` 只在 Flat 里启用，Rough 中权重为 `0.0`。这样 Rough 楼梯训练不会受影响，但也不能约束 Rough 直走时的趴低动作。

这次目标是在 Rough 中加入一个**弱的、只对直走生效的姿态约束**：

- 直走时，不鼓励把 base 压低到 `0.43` 以下。
- 转向时不启用，避免影响 yaw 控制。
- 爬楼梯/复杂地形中需要姿态变化时，尽量不强行锁死身体高度。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 扩展 `commanded_base_height_below_target`

给函数增加两个可选参数：

```python
straight_command_only: bool = False
max_abs_yaw_command: float | None = None
```

默认值保持旧行为不变，所以 Flat 的逻辑不受影响。

当 `straight_command_only=True` 时，函数只在下面条件满足时生效：

```python
command_xy_norm > command_threshold
yaw_command_abs <= max_abs_yaw_command
```

含义：

- 有前进/后退速度命令。
- yaw 命令很小，也就是近似直走。
- 这时如果 base 高度低于目标高度才扣分。

#### 在 Rough 中启用弱约束

在 JK03 Rough 配置中加入：

```python
commanded_base_height_below_target.weight = -0.35
height_margin = 0.10
command_threshold = 0.08
straight_command_only = True
max_abs_yaw_command = 0.08
```

解释：

- `-0.35` 比 Flat 的 `-1.2` 弱很多，避免过度限制爬楼梯。
- `height_margin = 0.10` 比 Flat 的 `0.08` 更宽松，允许 Rough 地形上有一点姿态变化。
- `straight_command_only=True` 保证只管直走，不管明显转向。
- `max_abs_yaw_command=0.08` 表示 yaw 命令很小时才认为是直走。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改楼梯 terrain generator。
- 没改变 Flat 上一版新增的 yaw reward 权重。

### 验证

- 本地 Python 编译检查通过。
- 云端已覆盖上传到 `/root/dog-robot-main/robot_lab-main`。
- 云端 Python 编译检查通过：`REMOTE_COMPILE_OK`。
- 云端确认 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)` 仍在原位。
- 云端确认 JK03 rough 无 terrain override：`NO_JK03_TERRAIN_OVERRIDE`。
- 待 GitHub commit/push。

### 已知风险

- 如果后续发现 Rough 直走变稳但爬楼梯更僵硬，说明 `-0.35` 仍然偏强，需要降到 `-0.15 ~ -0.25`。
- 如果直走仍然明显塌低，说明 `-0.35` 偏弱，可以小幅提高到 `-0.5`，但不建议直接用 Flat 的 `-1.2`。

### 下一步观察指标

- Rough TensorBoard：
  - `Episode_Reward/commanded_base_height_below_target`
  - `Episode_Reward/base_height_l2`
  - `Episode_Reward/stair_upward_progress`
  - `Episode_Reward/commanded_motion_progress`
  - `Metrics/base_velocity/error_vel_xy`
  - `Metrics/base_velocity/error_vel_yaw`
- 视频/键盘：
  - 直走时 base 是否还明显下沉。
  - 转向时是否仍能转。
  - 爬楼梯是否变僵硬。

## 2026-06-16: flat-basic-keyboard-control-v1

状态：本地最新版。目标是先把平地基础动作调顺，再继续 rough/stair。

### 为什么修改

用户在平地键盘测试中观察到：

- 按前进/后退时四肢关节明显弯曲，重心降低，像是靠趴低来移动。
- 左右转向很不灵敏，甚至几乎不能用键盘控制 yaw。
- 当前目标应先让 JK03 在 flat 上学会干净的直走、后退和左右转，再继续爬楼梯。

上一版 flat 虽然有 `track_ang_vel_z_exp` 和 `yaw_stuck_with_command`，但缺少两个更直接的基础信号：

- yaw 命令下，真的朝命令方向产生角速度。
- 有速度命令时，不能把 base 压低到目标高度以下来偷分。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`

### 怎么修改

#### 新增 `yaw_command_progress`

位置：`mdp/rewards.py`

逻辑：

```python
signed_yaw_rate = sign(command_yaw) * root_ang_vel_b_z
reward = clamp(signed_yaw_rate / max_yaw_rate, 0, 1)
```

含义：

- 左转命令时，只有真的左转才给分。
- 右转命令时，只有真的右转才给分。
- 原地不转、转反方向都不给这个奖励。

JK03 flat 中启用：

```python
yaw_command_progress.weight = 1.5
command_threshold = 0.06
max_yaw_rate = 0.35
```

目的：解决键盘左右转时策略没有足够直接动力去产生 yaw 的问题。

#### 新增 `commanded_base_height_below_target`

位置：`mdp/rewards.py`

逻辑：

```python
height_deficit = clamp(target_height - root_pos_w_z, min=0)
penalty = (height_deficit / height_margin) ** 2
```

含义：

- 只惩罚低于目标高度，不惩罚略高。
- 只在有 x/y/yaw 速度命令时启用。
- 如果按前进/后退/左右转时 base 往下塌，就会扣分。

JK03 flat 中启用：

```python
commanded_base_height_below_target.weight = -1.2
target_height = 0.43
height_margin = 0.08
command_threshold = 0.06
```

目的：抑制“按键后四肢弯曲、重心降低”的动作，让平地策略先学会站高移动。

#### Flat 训练改为直接 yaw-rate 命令

位置：`jk03/flat_env_cfg.py`

```python
self.commands.base_velocity.heading_command = False
```

原因：

- 键盘 play 发送的是直接速度命令：`vx, vy, yaw_rate`。
- Flat 预训练如果仍以 heading target 为主，和键盘测试的 direct yaw-rate 分布不完全一致。
- 先在 flat 上训练 direct yaw-rate，更适合验证左右键是否能直接转。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改楼梯 terrain generator 和 rough stair reward 权重。

### 验证

- 已进行本地 Python 编译检查。
- 云服务器当前旧端口连续 `Connection refused`，本次无法上传云端验证；等用户提供新的 SSH 端口后再同步。

### 已知风险

- `commanded_base_height_below_target` 如果权重过大，可能让策略过度保守、速度变慢。
- `yaw_command_progress` 会鼓励更主动转向，早期训练可能出现原地扭动，需要观察 `error_vel_yaw`、`track_ang_vel_z_exp` 和键盘 play 视频。
- Flat 改为 direct yaw-rate 后，后续 Rough 是否也同步改 direct yaw-rate，需要先看 Flat 效果再决定。

### 下一步观察指标

- Flat TensorBoard：
  - `Episode_Reward/yaw_command_progress`
  - `Episode_Reward/commanded_base_height_below_target`
  - `Metrics/base_velocity/error_vel_xy`
  - `Metrics/base_velocity/error_vel_yaw`
  - `Episode_Reward/track_lin_vel_xy_exp`
  - `Episode_Reward/track_ang_vel_z_exp`
- 视频/键盘：
  - 按前进时 base 是否还明显下沉。
  - 左右键是否能原地或低速稳定转向。
  - 后退是否比之前有响应。

## 2026-06-15: strict-net-stair-progress-v2

状态：当前最新版，已上传云服务器。需要重新启动训练进程才会生效。

### 为什么修改

用户实测发现上一版虽然 TensorBoard 里 `stair_upward_progress` 很高，但实际狗仍然不能稳定爬楼梯。说明旧的楼梯奖励可以被“假爬楼梯”刷分，例如：

- 身体向上颠一下。
- 轮子顶住台阶边后被挤高。
- 在坡面或台阶边获得 z 方向上升，但没有真正一级一级爬。

所以这次目标不是继续加大奖励，而是让楼梯奖励更难作弊。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`

### 怎么修改

#### `stair_upward_progress`

旧逻辑偏宽松：只要短窗口内有一些向前和 z 方向上升，就能得到较高楼梯奖励。

新逻辑改为净位移判断：

- 使用 `window_steps = 30` 的长窗口。
- 必须满足净向前距离超过 `min_forward_step = 0.08`。
- 必须满足净高度上升超过 `min_up_step = 0.025`。
- 用 `max_forward_step = 0.30` 和 `max_up_step = 0.09` 做归一化。
- 主要奖励 `forward_progress * upward_progress` 的耦合项。
- 参数：

```python
max_forward_step = 0.30
max_up_step = 0.09
min_forward_step = 0.08
min_up_step = 0.025
window_steps = 30
forward_weight = 0.0
coupled_weight = 0.85
upward_weight = 0.15
min_forward_fraction = 0.30
```

含义：只有真的向前跨出一段距离，并且净高度上升，才更容易拿到楼梯奖励。

#### 新增 `upward_without_forward_progress`

新增反作弊惩罚：

```python
upward_without_forward_progress.weight = -1.0
```

如果机器人 z 方向上升了，但 x 方向净向前不足 `0.08`，就扣分。目的是抑制：

- 原地往上蹦。
- 身体顶台阶边被挤高。
- 只抬身体、不往前跨。

#### 保留和调整的辅助奖励

```python
wheel_clearance_on_command.weight = 1.6
vertical_bounce_without_progress.weight = -0.35
wheel_spin_without_progress.weight = -0.05
wheel_lateral_edge_contact.weight = -0.06
feet_stumble.weight = -0.35
feet_slide.weight = -0.16
```

解释：

- `wheel_clearance_on_command` 保留抬轮信号，避免重新回到贴地走。
- `vertical_bounce_without_progress` 抑制没有前进的上下弹跳。
- `wheel_spin_without_progress` 抑制轮子空转。
- `wheel_lateral_edge_contact` 抑制轮子侧边顶台阶。
- `feet_stumble` 和 `feet_slide` 控制绊脚和滑动，但权重没有调得过大，避免再次压死抬腿。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py`。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。

### 验证

- 本地 `python3 -m py_compile` 通过。
- 云端 `python -m py_compile` 通过。
- 云端确认：

```text
terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
NO_JK03_TERRAIN_OVERRIDE
```

### 已知风险

- `stair_upward_progress` 早期会明显变低，这是预期，因为它不再容易被刷分。
- `terrain_levels` 前期可能涨得更慢。
- 如果到 `500 step` 后 `terrain_levels < 1` 且 `commanded_motion_progress` 仍很低，说明阈值可能太严格，需要考虑放松 `min_forward_step` 或 `min_up_step`。

### 下一步观察

重点看窗口均值，不看单点：

- `commanded_motion_progress` 是否从 `0.02` 往 `0.06-0.10` 上升。
- `stair_upward_progress` 是否从接近 `0` 稳定上升。
- `wheel_clearance_on_command` 是否继续提高。
- `terrain_levels` 是否能从 `1.x` 继续上升。
- `feet_slide`、`wheel_lateral_edge_contact`、`wheel_spin_without_progress` 是否恶化。

## 2026-06-15: smooth-coupled-stair-progress-v1

状态：已废弃。该版本指标好看，但用户实测仍不能爬楼梯。

### 为什么修改

旧版训练中 `stair_upward_progress` 长期卡在约 `0.07`，`wheel_clearance_on_command` 下降，狗更倾向保守走法，不能形成抬轮上台阶动作。

### 怎么修改

- `stair_upward_progress` 改成短窗口耦合奖励。
- 使用 `window_steps = 10`。
- 使用 `forward_progress * upward_progress` 和平方根平滑项。
- `wheel_clearance_on_command` 在低速时保留 `min_progress_scale = 0.35`，避免抬轮奖励被速度门槛完全卡死。
- 降低部分会压制抬腿的惩罚：
  - `lin_vel_z_l2`
  - `base_height_l2`
  - `feet_stumble`
  - `wheel_lateral_edge_contact`
  - `feet_height_body`

### 训练观察

该版初期 TensorBoard 改善明显：

- `stair_upward_progress` 曾从 `0.254` 到 `0.604`。
- `wheel_clearance_on_command` 曾从 `0.109` 到 `0.237`。
- `illegal_contact` 明显下降。

### 后续问题

用户实测仍不能爬楼梯，说明该指标存在刷分空间。可能原因：

- 机器人通过蹭边、顶台阶或身体被挤高获得 z 上升。
- 指标高不代表真实一级一级爬。
- `stair_upward_progress` 过于乐观。

处理：升级为 `strict-net-stair-progress-v2`。

## 2026-06-15: restore-fanziqi-terrain-curriculum

相关提交：

- `dff4d9d Restore JK03 terrain curriculum defaults`
- `e742346 Use base terrain curriculum for JK03 rough`

### 为什么修改

训练中 `terrain_levels` 长期卡住或出现异常波动。用户明确要求 terrain level 算法恢复为 fan-ziqi 原始基类算法，不再自定义。

### 怎么修改

- 恢复 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
- 删除 JK03 rough 内的自定义 terrain override。
- 不再使用自定义 `terrain_levels_jk03_stairs` 作为 JK03 rough curriculum。

### 没有修改

- 没改 reward。
- 没改 URDF。
- 没改 `jk03.py`。

### 后续约束

以后不能再修改 terrain curriculum 算法。楼梯训练问题只能通过 reward、PPO、命令范围、测试方式解决。

## 2026-06-15: tightened-jk03-stair-curriculum

相关提交：

- `23fd776 Tighten JK03 rough stair curriculum`

状态：已回退。

### 为什么修改

当时 `terrain_levels` 出现卡住和实际爬楼梯差的问题，尝试通过更慢、更适合楼梯的 curriculum 判断来改善。

### 后续问题

用户确认不希望改变 fan-ziqi 原始 terrain level 算法。该方向已停止，并通过 `restore-fanziqi-terrain-curriculum` 回退。

## 2026-06-15: stair-progress-reward-family

相关提交：

- `4e888b9 Add JK03 stair progress rewards`

### 为什么修改

JK03 在 rough/stair 场景下容易轮子空转、顶台阶、身体被卡住，但速度 tracking reward 仍可能看起来不错。需要加入真实位移和楼梯进展相关奖励。

### 怎么修改

新增或强化了一组 reward：

- `commanded_motion_progress`：奖励沿命令方向的真实位移。
- `stair_upward_progress`：奖励向前同时上升。
- `wheel_spin_without_progress`：惩罚有命令但无位移时轮子空转。
- `wheel_lateral_edge_contact`：惩罚轮子侧向顶住台阶边。
- `wheel_clearance_on_command`：鼓励移动时抬轮。

### 后续问题

这些奖励方向是合理的，但早期版本容易出现：

- 指标虚高。
- 真实爬楼梯能力弱。
- 轮子蹭边或滑动增加。

因此后续逐步收紧 `stair_upward_progress` 的定义。

## 2026-06-15: ppo-stability-tuning

相关提交：

- `280f9d9 Stabilize JK03 PPO training`
- `869cd8f Rebalance JK03 PPO for stable motion`

### 为什么修改

训练中出现策略不稳定、动作幅度大、转向动作奇怪、加载中断 checkpoint 后表现退化等问题。

### 修改方向

- 降低 PPO 更新过猛的风险。
- 调整 learning rate / entropy / noise / rollout 相关配置。
- 让 rough stair 训练更稳，减少策略突然学坏。

### 后续观察

PPO 指标重点看：

- `value_loss`
- `surrogate_loss`
- `entropy_loss`
- `mean_noise_std`
- `mean_reward`
- `mean_episode_length`

如果 reward 指标改善但视频行为差，优先怀疑 reward 被刷分，而不是只调 PPO。

## 2026-06-13 to 2026-06-15: keyboard-and-play-debug

相关提交：

- `c6705bc Document JK03 keyboard troubleshooting`
- `2242762 Stabilize JK03 keyboard play spawn`
- `7e5aa75 Stabilize JK03 rough yaw control`

### 为什么修改

用户在 Isaac Sim play 模式下遇到：

- 键盘输入没有反应。
- 终端没有 keyboard 输出。
- 有 keyboard 输出但狗不动。
- 左右转时腿大幅异常摆动。
- 初始位置可能卡在地形缝里。

### 修改方向

- 给 `play.py` 增加 keyboard/debug 输出。
- 修复 play spawn/初始位置相关问题。
- 记录 keyboard troubleshooting 到 README。
- 调整 yaw 控制相关 reward，改善转向卡住。

### 后续问题

如果 keyboard 有输出但狗不动，通常说明：

- policy 没学会响应该命令；
- checkpoint 太早或训练退化；
- spawn 点卡住；
- reward 对该命令方向覆盖不足。

## 2026-06-13: rough-stair-reward-tuning

相关提交：

- `263132c Tune JK03 rough stair rewards`
- `4a50060 Improve JK03 stair gait stability`
- `63e8816 Align JK03 rough config with wheeled baselines`

### 为什么修改

JK03 重约 50 kg，比常见 15-20 kg 机器狗更难训练。早期 rough 训练中出现：

- 后腿内八。
- 轮子贴台阶边滑上去，而不是抬着上。
- 平地转向姿态异常。
- 楼梯上不丝滑，靠碰撞和轮子滚。

### 修改方向

- 调整关节偏离惩罚。
- 调整 gait 和 symmetry。
- 调整轮子空转、侧向力、滑动相关惩罚。
- 参考 wheeled baseline 配置，但不照搬轻量机器狗参数。

### 后续问题

重机器狗对 reward 比例更敏感。过强的平稳惩罚会压死抬腿，过强的上楼梯奖励又容易导致蹭边和滑动。

## 2026-06-09 to 2026-06-12: cloud-training-and-compatibility

相关提交：

- `d4bf6f9 Adapt IsaacLab wrapper for RSL-RL TensorDict API`
- `b674772 Handle missing IsaacLab pretrained checkpoint helper`
- `319fd1d Document JK03 testing and training workflow`
- `25cc4a9 Switch JK03 README tests to rough task`
- `d7499ab Document JK03 resume training workflow`

### 为什么修改

云服务器使用 Isaac Sim / Isaac Lab / RSL-RL 版本和本地不同，早期出现：

- `RslRlBaseRunnerCfg` 导入失败。
- `handle_deprecated_rsl_rl_cfg` 导入失败。
- `omni.log` 缺失。
- checkpoint/resume/play 命令不清楚。
- TensorBoard / video / keyboard 测试流程混乱。

### 修改方向

- 兼容新的 Isaac Lab / RSL-RL API。
- 补充 README 训练、测试、恢复训练、TensorBoard 查看流程。
- 明确 rough 任务命令。

## 训练监测口径

以后报告训练结果时，必须看窗口均值趋势，不看单点。

默认窗口：

- 每 `250 step` 一个窗口。
- 对比当前窗口均值、上一窗口均值、差值和百分比变化。
- 尽量看最近 3 个窗口方向。

核心指标：

- `terrain_levels`：是否升难度。
- `commanded_motion_progress`：是否真实向命令方向移动。
- `stair_upward_progress`：是否真实向前并净上升。
- `wheel_clearance_on_command`：是否学会抬轮。
- `upward_without_forward_progress`：是否存在假上升。
- `illegal_contact`：是否乱撞。
- `feet_slide`：是否滑动。
- `wheel_lateral_edge_contact`：是否蹭台阶边。
- `wheel_spin_without_progress`：是否空转。
- `value_loss` / `surrogate_loss` / `entropy_loss`：PPO 是否稳定。

如果实际视频和 TensorBoard 冲突，以实际视频和真实行为为准，回头修正 reward 定义。
