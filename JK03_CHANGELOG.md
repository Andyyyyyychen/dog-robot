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
