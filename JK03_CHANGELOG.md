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

## 2026-06-22: wheel-yaw-open-loop-diagnostic-v10

状态：新增 JK03 开环轮子差速 yaw 诊断工具；不修改训练 reward、不修改 PPO、不修改 JK03 参数。

### 为什么修改

连续多版训练后，用户实测仍然出现：

- 按 `Z/X` 只能拧 hipx 关节，机身最多转约 45 度后卡住。
- 修改 reward 后无法稳定判断是“policy 没学会 yaw”，还是“轮子/摩擦/动作映射本身无法通过差速产生 yaw”。

因此先增加一个不经过 policy 的物理测试：直接给四个轮子施加开环速度动作，左轮和右轮反向转，看机身是否真实产生 yaw 角速度和 yaw 角位移。

### 修改文件

- `robot_lab-main/scripts/tools/check_jk03_wheel_yaw.py`
- `JK03_CHANGELOG.md`

### 怎么修改

新增 `check_jk03_wheel_yaw.py`：

- 加载 Isaac Lab / RobotLab 环境，默认任务为 `RobotLab-Isaac-Velocity-Flat-JK03-v0`。
- 将 `num_envs` 固定为诊断用的 1 个环境。
- 关闭观测噪声、外力、push、质量/材质随机化，降低开环测试中的随机因素。
- 如果加载 rough 任务，会关闭 terrain curriculum 并将 terrain 限制到最低等级，避免地形影响轮子差速判断。
- 每个 phase 先用零 action 稳定一小段时间，再施加固定轮子 action。
- 默认假设最后 4 个 action 是 `[FL, FR, RL, RR]` 轮子速度动作。
- 测试两组 yaw：
  - `yaw_left`: `[+1, -1, +1, -1] * wheel_action`
  - `yaw_right`: `[-1, +1, -1, +1] * wheel_action`
- 输出每组测试的：
  - `yaw_delta`：机身 yaw 角变化，单位 degree。
  - `mean_wz`：机身平均 yaw 角速度，单位 rad/s。
  - `mean_vx / mean_vy`：机身平均前后/左右速度。
  - `PASS / FAIL`：是否超过最小 yaw 角度或 yaw 角速度阈值。

### 如何运行

云服务器上建议先停止当前 play 或训练，再运行，避免两个 Isaac Sim 同时抢 GPU：

```bash
cd /root/dog-robot-main/robot_lab-main

/root/IsaacLab/isaaclab.sh -p scripts/tools/check_jk03_wheel_yaw.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 1 \
  --wheel_action 1.0 \
  --duration 4.0
```

如果 yaw 反应很弱，可以提高轮子动作幅度做诊断：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/tools/check_jk03_wheel_yaw.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 1 \
  --wheel_action 2.0 \
  --duration 4.0 \
  --include_forward
```

### 结果怎么判断

- 如果 `yaw_left` 和 `yaw_right` 都是 `FAIL`，说明开环轮子差速也几乎转不动，这时继续调 PPO/reward 意义很小，要优先检查轮子方向、接触摩擦、轮子驱动力、动作尺度或四个轮子的 action 顺序。
- 如果 `yaw_left` 和 `yaw_right` 都能转，但方向相同，说明轮子 action 顺序或正负号假设有问题。
- 如果 `yaw_left` 和 `yaw_right` 能朝相反方向转，说明物理层面可以差速 yaw，后续再回到 command mapping、reward 权重和 PPO 训练上修改。

### 没有修改什么

- 没有修改 `jk03.py`。
- 没有修改 JK03 URDF。
- 没有修改 terrain curriculum 算法。
- 没有修改 reward 权重。
- 没有修改 PPO 配置。

### 已知风险

- 脚本默认最后 4 个 action 是轮子速度动作。如果后续动作顺序改变，测试结论会失效。
- 脚本默认轮子顺序按 `[FL, FR, RL, RR]` 解释；如果实际 action 顺序不同，`yaw_left/yaw_right` 的方向判断需要按输出重新校准。
- 开环测试不是训练效果评估，只用于判断底层物理和动作映射是否支持轮式差速 yaw。

### 2026-06-22 补充修复

第一次云端运行时，`yaw_left` 已经测到 `yaw_delta=-23.37 deg`、`mean_wz=-0.3527 rad/s`，说明 JK03 在物理层面可以靠轮子差速产生 yaw。但脚本在第二个 phase reset 时触发 PyTorch `inference_mode` 与 Isaac Lab 内部 reset 写张量冲突。

修复：

- 将测试循环从 `torch.inference_mode()` 改成 `torch.no_grad()`，避免环境内部状态张量被标记成 inference tensor。
- 诊断环境额外关闭 `randomize_rigid_body_mass_base`、`randomize_rigid_body_mass_others`、`randomize_com_positions`、`randomize_push_robot`、`randomize_reset_joints`、`randomize_actuator_gains`，减少开环测试中的随机因素。

这次仍然没有修改 `jk03.py`、URDF、terrain curriculum、reward 权重或 PPO 配置。

## 2026-06-22: mature-wheeled-baseline-v9

状态：本地静态编译通过；目标是停止继续叠加手写 yaw 奖励，回到成熟轮足基线，让 flat 先恢复稳定移动和标准 yaw 速度跟踪。

### 为什么修改

连续几版为了修复转向，加入了 yaw 专用奖励、轮速差奖励、抬轮/对角踏步奖励和 hipx 约束。实际训练和视频表现说明这些奖励互相抢目标：

- `yaw_wheel_differential_progress` / `yaw_wheel_velocity_alignment` 容易奖励“轮子有速度差”，但不一定奖励“机身真的稳定转过去”。
- 过强的 `yaw_stuck_with_command`、抬轮相位奖励、脚滑惩罚会让早期策略宁愿不动，或者用拧 hipx 的方式拿局部收益。
- `feet_slide` 对轮式足端不适合直接照搬普通足式机器人，因为轮子接地滚动时足端相对机身本来就有切向速度，惩罚太大容易压住轮式运动。
- `track_lin_vel_xy_exp` 和 `track_ang_vel_z_exp` 过大时，会把策略推向快速拿 tracking 奖励的捷径，不一定产生健康姿态。

### 参考基线

对照 RobotLab 中成熟轮足配置：

- `unitree_go2w`
- `unitree_b2w`
- `deeprobotics_m20`

这些配置的共同点是：

- 轮子速度动作 scale 通常使用 `5.0`。
- yaw 主要靠标准 `track_ang_vel_z_exp`，而不是大量手写 yaw 奖励。
- 轮式足端的 `feet_slide` 通常为 `0`。
- 腿部姿态主要用 `joint_pos_penalty` 和少量 joint deviation 约束。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

Flat：

- `joint_vel.scale: 8.0 -> 5.0`，降低轮子动作幅度，避免转向时轮速过冲。
- `track_lin_vel_xy_exp: 9.0 -> 5.0`，降低直行奖励对总 reward 的压制。
- `track_ang_vel_z_exp: 1.6 -> 1.8`，保留 yaw 跟踪，但不再依赖手写 yaw 轮速奖励。
- `yaw_command_progress: 0.55 -> 0`。
- `yaw_wheel_differential_progress: 0.25 -> 0`。
- `yaw_hipx_twist_without_yaw_progress: -3.0 -> 0`。
- `joint_deviation_hipx_l1: -0.55 -> -0.80`。
- `joint_pos_penalty: -0.90 -> -1.00`。
- `yaw_turn_joint_posture_l2: -0.30 -> -0.45`。
- `yaw_stuck_with_command: -6.0 -> -2.0`，避免早期因为 yaw 不足被过度惩罚到不敢探索。
- yaw command range 从 `+-0.9` 收到 `+-0.6`，先学可实现的稳定转向，再扩大范围。

FlatYaw：

- yaw command range 从 `+-0.9` 收到 `+-0.6`。
- `track_ang_vel_z_exp: 1.8 -> 2.0`。
- 关闭 `yaw_command_progress` 和 `yaw_wheel_differential_progress`。
- `joint_deviation_hipx_l1: -0.80 -> -0.90`。
- `joint_pos_penalty: -1.00 -> -1.10`。
- `yaw_turn_joint_posture_l2: -0.30 -> -0.55`。

Rough：

- `joint_vel.scale: 8.0 -> 5.0`。
- `track_lin_vel_xy_exp: 8.0 -> 5.0`。
- `track_ang_vel_z_exp: 4.0 -> 1.5`。
- `joint_deviation_hipx_l1: -0.16 -> -0.25`。
- `joint_pos_penalty: -0.45 -> -0.80`。
- `feet_slide: -0.16 -> 0`。
- `feet_gait: 0.22 -> 0`。
- 楼梯相关奖励整体降温：`stair_upward_progress 3.0 -> 2.0`、`wheel_clearance_on_command 1.6 -> 0.8` 等。

### 没有修改什么

- 没有修改 `jk03.py`。
- 没有修改 JK03 URDF。
- 没有修改 terrain curriculum 算法。
- 没有修改 PPO 结构。

### 预期效果

这版不再强行教“对角抬腿转向”。目标先回到更成熟的轮足基线：

- 前进/后退/左右平移先稳定。
- yaw 转向尽量靠标准角速度 tracking 学出来。
- hipx 拧腿由更普通、更稳定的姿态惩罚压住。
- rough 训练不再过早被楼梯/抬轮奖励带偏。

### 后续判断

新训练必须重新启动，旧进程不会自动加载这版代码。验证时优先看：

- `error_vel_yaw` 是否下降。
- `track_ang_vel_z_exp` 是否上升。
- `joint_deviation_hipx_l1` 是否没有继续变大。
- `joint_pos_penalty` 是否没有持续恶化。
- 实际视频里是否还靠 hipx 拧到 45 度后卡住。

## 2026-06-22: flat-yaw-mature-wheel-baseline-v8

状态：本地静态编译通过；目标是修正 v7 中“轮差速奖励上升，但实际 yaw 仍弱，并且继续拧 hipx 关节”的问题。

### 为什么修改

用户实测反馈：

- 狗又回到拧关节转向。
- Z/X yaw 仍然不能稳定持续转，转到一定角度就卡住。

云端 TensorBoard 最新 run：`jk03_flat/2026-06-22_10-00-12`，latest step `762`。最近 3 个 200-step 窗口趋势显示：

- `mean_reward`: `114.55 -> 124.62 -> 137.32`，总体奖励在上升。
- `yaw_wheel_velocity_alignment`: `0.892 -> 1.117 -> 1.284`，v7 新增的轮速方向奖励明显上升。
- `yaw_wheel_differential_progress`: `0.177 -> 0.364 -> 0.500`，轮差速相关奖励也上升。
- 但 `error_vel_yaw`: `0.861 -> 0.751 -> 0.845`，当前窗口反而变差。
- `track_ang_vel_z_exp`: `1.232 -> 1.335 -> 1.264`，当前窗口下降。
- `feet_slide`: `-1.128 -> -1.172 -> -1.249`，脚滑/轮滑变差。
- `joint_deviation_hipx_l1`: `-0.106 -> -0.150 -> -0.174`，hipx 偏离越来越大。

判断：v7 的 `yaw_wheel_velocity_alignment` 确实教会了轮子做差速，但没有保证身体真实 yaw 跟上。策略利用了“轮子差速 + hipx 拧腿”的局部最优，导致指标上轮差速奖励很好看，实际表现仍然是拧关节。

参考成熟轮足配置：

- `unitree_go2w`、`unitree_b2w`、`deeprobotics_m20` 都使用较小轮速动作：`joint_vel.scale = 5.0`。
- 它们的 `track_ang_vel_z_exp.weight` 通常为 `1.5`，不是很大。
- 它们对腿部姿态用较强 `joint_pos_penalty.weight = -1.0`。
- 它们对 wheeled foot 的 `feet_slide.weight = 0`，因为轮足机器人轮子在接触中滚动，直接用足端滑动惩罚容易误伤转向。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 新增 `yaw_hipx_twist_without_yaw_progress`

新增 reward 函数：

```python
yaw_hipx_twist_without_yaw_progress(...)
```

逻辑：

- 只在 yaw 命令明显时启用。
- 读取 hipx 关节相对默认站姿的偏离。
- 读取真实 body yaw rate。
- 如果 hipx 偏离大，但真实 yaw rate 没有达到阈值，就惩罚。
- 如果身体已经真实转起来，则这个惩罚自动减弱。

含义：不是简单禁止 hipx 动，而是禁止“拧 hipx 但身体不转”的坏动作。

#### 2. 注册新 reward

在 `JK03RewardsCfg` 中新增：

```python
yaw_hipx_twist_without_yaw_progress = RewTerm(...)
```

默认权重为 `0.0`，只在 flat / flat-yaw 中打开。

#### 3. 关闭 v7 的误导性轮速方向奖励

普通 flat：

```python
yaw_wheel_velocity_alignment.weight: 2.00 -> 0
```

Flat-Yaw：

```python
yaw_wheel_velocity_alignment.weight: 2.50 -> 0
```

原因：报告证明它上升时，真实 yaw 跟踪没有同步上升，反而鼓励轮子差速/空转。

#### 4. 降低轮差速 reward

普通 flat：

```python
yaw_wheel_differential_progress.weight: 1.00 -> 0.25
```

Flat-Yaw：

```python
yaw_wheel_differential_progress.weight: 1.20 -> 0.35
```

含义：保留一点“轮差速辅助信号”，但不再让它主导策略。

#### 5. 回到成熟轮足的轮速和 yaw 权重尺度

普通 flat：

```python
joint_vel.scale: 16.0 -> 8.0
track_ang_vel_z_exp.weight: 2.4 -> 1.6
yaw_command_progress.weight: 0.55
```

Flat-Yaw：

```python
track_ang_vel_z_exp.weight: 2.8 -> 1.8
yaw_command_progress.weight: 0.65
```

含义：避免过强轮速动作和过强 yaw 奖励把策略推向“高速轮转 + hipx 拧腿”的捷径。

#### 6. 加强姿态稳定，关闭 feet_slide

普通 flat：

```python
joint_pos_penalty.weight: -0.25 -> -0.90
joint_deviation_hipx_l1.weight: -0.25 -> -0.55
feet_slide.weight: -0.70 -> 0
action_rate_l2.weight: -0.002 -> -0.01
yaw_hipx_twist_without_yaw_progress.weight = -3.0
```

Flat-Yaw：

```python
joint_pos_penalty.weight: -0.25 -> -1.00
feet_slide.weight: -0.55 -> 0
yaw_hipx_twist_without_yaw_progress.weight = -3.5
```

含义：

- 像成熟轮足配置一样，让腿部姿态正则更强。
- 对轮足机器人先关闭 `feet_slide`，避免把轮子滚动误判成脚滑。
- 用 targeted penalty 专门处理“拧 hipx 但 yaw 不动”。

### 没有修改什么

- 没有修改 `jk03.py` 中任何 JK03 初始物理参数。
- 没有修改 JK03 URDF。
- 没有修改 fan-ziqi 原始 terrain curriculum 算法。
- 没有修改 rough 的 terrain level 计算逻辑。
- 没有恢复强制抬腿/对角踏步奖励。

### 验证结果

- 本地 `python3 -m py_compile` 通过：
  - `rewards.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
- 受保护文件 diff 为空：
  - `jk03.py`
  - `jk03.urdf`
  - `velocity_env_cfg.py`
  - `curriculums.py`

### 已知风险

- 关闭 `feet_slide` 后，早期可能看不到脚滑指标约束，但这是参考成熟 wheeled 配置后的取舍。
- 轮速动作从 `16.0` 降到 `8.0` 后，转向可能更慢，但应减少空转和姿态扭曲。
- 如果仍然拧 hipx，下一步不应再调 reward，而应该做开环测试：直接给轮子速度命令，看仿真物理上能否原地 yaw。

### 下一步观察指标

- `error_vel_yaw` 是否下降。
- `track_ang_vel_z_exp` 是否上升或至少稳定。
- `joint_deviation_hipx_l1` 是否下降。
- `yaw_hipx_twist_without_yaw_progress` 是否从高值下降。
- `yaw_stuck_with_command` 是否继续下降。
- play 中按 Z/X 时 `base_wz` 是否持续非零，而不是 hipx 拧到一定角度后停止。

## 2026-06-22: flat-yaw-wheel-alignment-v7

状态：本地静态编译通过；目标是解决 flat 测试中“Z/X yaw 最多拧到约 45 度后卡住，不能持续原地转”的问题。

### 为什么修改

用户实测反馈：

- 现在 JK03 仍然不能稳定原地转向。
- yaw 时最多靠关节/机身姿态拧一下，转到约 45 度后卡住。
- 说明上一版虽然加强了 yaw 奖励和 hipx 姿态约束，但策略仍然没有学到“用左右轮差速持续制造 yaw”的基础动作。

判断原因：

- `yaw_wheel_differential_progress` 需要“左右轮有差速，并且身体已经按命令方向转起来”才给奖励。训练早期如果身体还没有转起来，这个奖励太稀疏。
- 普通 flat 里的 yaw 专项奖励只在 xy 命令很小的时候启用，随机训练中纯 yaw 样本比例较低，导致键盘 Z/X 对应的原地 yaw 学得不够。
- `yaw_turn_joint_posture_l2` 太强时会把 yaw 探索压住；太弱时又会靠 hipx 拧腿。本版改为用轮速方向奖励主导 yaw，而不是继续堆 hipx 惩罚。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 新增 `yaw_wheel_velocity_alignment`

新增 reward 函数：

```python
yaw_wheel_velocity_alignment(...)
```

逻辑：

- 读取 yaw 命令 `command[:, 2]`。
- 读取四个轮子的关节速度。
- 计算左侧轮平均速度：`fl + hl`。
- 计算右侧轮平均速度：`fr + hr`。
- 对正 yaw 命令，奖励 `right_wheel_vel - left_wheel_vel` 为正。
- 对负 yaw 命令，奖励 `right_wheel_vel - left_wheel_vel` 为负。
- 这个奖励不要求身体已经转起来，只要求轮子先做出正确方向的差速。

含义：先教会策略“Z/X yaw 时轮子该怎么转”，再由 `yaw_wheel_differential_progress` 和 `track_ang_vel_z_exp` 要求身体真的跟着转。

#### 2. 注册新 reward

在 `JK03RewardsCfg` 中新增：

```python
yaw_wheel_velocity_alignment = RewTerm(...)
```

默认权重为 `0.0`，只在 flat / flat-yaw 里打开。

#### 3. 普通 flat 中打开轮速方向奖励

普通 `JK03FlatEnvCfg`：

```python
yaw_wheel_velocity_alignment.weight = 2.00
yaw_wheel_velocity_alignment.max_xy_command = 1.20
yaw_wheel_velocity_alignment.target_wheel_diff = 5.0
```

含义：不只在完全原地 yaw 时给轮差速方向奖励，只要 yaw 命令存在且 xy 命令没有超过训练范围，就给策略更密集的转向学习信号。

#### 4. 调整 yaw 相关权重

普通 flat：

```python
track_ang_vel_z_exp.weight: 3.0 -> 2.4
yaw_command_progress.weight: 0.85 -> 0.75
yaw_wheel_differential_progress.weight: 1.20 -> 1.00
yaw_wheel_differential_progress.max_xy_command: 0.16 -> 1.20
yaw_stuck_with_command.weight: -6.0 -> -8.0
```

含义：

- 不继续单纯加大身体 yaw 奖励，避免策略靠拧关节骗一点 yaw。
- 增强“有 yaw 命令但身体不转”的惩罚。
- 让轮差速 reward 在更多 yaw 样本中生效。

#### 5. 放松会卡死 yaw 探索的姿态惩罚

普通 flat：

```python
yaw_turn_joint_posture_l2.weight: -1.20 -> -0.45
joint_pos_penalty.weight: -0.35 -> -0.25
joint_deviation_hipx_l1.weight: -0.22 -> -0.25
```

含义：

- `yaw_turn_joint_posture_l2` 从强约束改成温和约束，避免一边要求 yaw、一边把所有 yaw 探索压死。
- `joint_deviation_hipx_l1` 稍微加强，继续压住大幅 hipx 拧腿。
- `joint_pos_penalty` 略微放松，让腿部能配合轮子做必要支撑，而不是僵住。

#### 6. 适当放开轮子速度动作

普通 flat：

```python
self.actions.joint_vel.scale: 12.0 -> 16.0
```

含义：JK03 质量较大，原地 yaw 需要更明显的左右轮速度差。轮子速度动作太小会导致策略即使想转也推不动。

#### 7. 调整脚滑惩罚

普通 flat：

```python
feet_slide.weight: -0.85 -> -0.70
yaw_slide_scale: 0.25 -> 0.15
```

含义：直线和平移仍然惩罚滑动，但 yaw 时允许必要的轮/脚切向运动，避免转向刚开始就被脚滑惩罚压住。

#### 8. Flat-Yaw 专项同步调整

`JK03FlatYawEnvCfg`：

```python
track_ang_vel_z_exp.weight: 3.5 -> 2.8
yaw_command_progress.weight: 1.00 -> 0.90
yaw_wheel_differential_progress.weight: 1.50 -> 1.20
yaw_wheel_velocity_alignment.weight = 2.50
yaw_stuck_with_command.weight: -7.0 -> -8.0
feet_slide.weight: -0.70 -> -0.55
yaw_slide_scale: 0.20 -> 0.15
joint_pos_penalty.weight: -0.35 -> -0.25
yaw_turn_joint_posture_l2.weight: -1.40 -> -0.45
```

含义：Flat-Yaw 仍然是专门练原地 yaw 的任务，但本版核心目标从“强制抬腿/强制姿态”改成“轮子先学会正确差速，身体再持续 yaw”。

### 没有修改什么

- 没有修改 `jk03.py` 中任何 JK03 初始物理参数。
- 没有修改 JK03 URDF。
- 没有修改 fan-ziqi 原始 terrain curriculum 算法。
- 没有修改 rough 的 terrain level 计算逻辑。
- 没有恢复强制抬腿/对角踏步奖励。

### 验证结果

- 本地 `python3 -m py_compile` 通过：
  - `rewards.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
- 受保护文件 diff 为空：
  - `jk03.py`
  - `jk03.urdf`
  - `velocity_env_cfg.py`
  - `curriculums.py`

### 已知风险

- 如果实际轮子正负方向和这里推断相反，`yaw_wheel_velocity_alignment` 会给错方向奖励；此时训练中 `yaw_wheel_velocity_alignment` 可能上升但 `track_ang_vel_z_exp` 不上升，需要把左右轮差速符号反过来。
- 轮子速度动作增大后，早期可能出现轮子转得更猛、`feet_slide` 短期变差；关键要看 `track_ang_vel_z_exp` 和 `yaw_stuck_with_command` 是否明显改善。
- 这一版仍然不是楼梯抬腿方案，目标是先让 flat 中 Z/X 可以持续 yaw，最好能原地持续转圈。

### 下一步观察指标

- `yaw_wheel_velocity_alignment` 是否快速变成稳定正值。
- `yaw_wheel_differential_progress` 是否随后上升。
- `track_ang_vel_z_exp` 是否上升。
- `yaw_stuck_with_command` 绝对值是否下降。
- `joint_deviation_hipx_l1` 是否没有明显变大。
- play 测试中 `base_wz` 是否能在按住 Z/X 时持续非零，而不是只转到约 45 度就停。

## 2026-06-22: flat-yaw-differential-v6

状态：本地静态编译通过；未修改 JK03 原始参数、URDF、terrain curriculum。

### 为什么修改

用户实测反馈：普通 flat 训练/测试时仍然不能正常原地转向，按 yaw 后最多把腿部/hipx 关节拧到约 45 度，然后卡住不再继续转。

判断原因：

- 原来的 yaw 奖励主要看身体 yaw 角速度，没有明确告诉策略“用左右轮反向差速来制造 yaw”。
- `yaw_stuck_with_command` 只惩罚没转起来，但没有奖励正确的轮子差速动作。
- `feet_slide` 在原地 yaw 时会把轮/脚绕身体中心产生的切向速度也算成滑动，导致“想转就被脚滑惩罚拉住”。
- 腿部 position action 没有在 flat 里收紧 clip，策略仍可能输出很大的 hipx 目标角，用拧腿去尝试转向。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 新增 `yaw_wheel_differential_progress`

新增 reward 函数：

```python
yaw_wheel_differential_progress(...)
```

逻辑：

- 只在 yaw 命令明显、xy 平移命令很小时启用。
- 读取 4 个轮子关节速度。
- 计算左侧轮平均速度和右侧轮平均速度的差值。
- 同时要求身体 yaw 角速度方向和命令方向一致。
- 只有“左右轮速度有差 + 身体真的按命令方向转起来”才给奖励。

含义：把原来的“你要转起来”变成更明确的“你要靠左右轮差速转起来”，减少靠 hipx 拧腿骗 yaw 的可能。

#### 2. `feet_slide` 增加 yaw 降权参数

给 `feet_slide` 增加可选参数：

```python
command_name
yaw_command_threshold
max_xy_command
yaw_slide_scale
```

逻辑：

- 普通前进、后退、左右平移时，脚滑惩罚照常生效。
- 当命令是接近原地 yaw 时，脚滑惩罚乘以 `yaw_slide_scale`。

含义：原地转向时轮子/脚相对身体有切向速度，这是差速转弯必然出现的运动，不能完全按直线行走的脚滑标准惩罚。

#### 3. flat 限制腿部 action clip

新增：

```python
self.actions.joint_pos.clip = {".*": (-1.0, 1.0)}
```

含义：腿部 position action 不再允许输出非常大的目标角。结合 hipx scale `0.06`，正常情况下 hipx 目标偏移会被限制在约 `+-0.06 rad`，避免继续出现“拧到 45 度左右卡住”的动作模式。

#### 4. flat 增强 yaw 真实转动奖励

普通 flat：

```python
track_ang_vel_z_exp.weight: 1.8 -> 3.0
yaw_command_progress.weight: 0.35 -> 0.85
yaw_wheel_differential_progress.weight: 0 -> 1.20
yaw_stuck_with_command.weight: -4.0 -> -6.0
```

Yaw 专项 flat：

```python
track_ang_vel_z_exp.weight: 1.2 -> 3.5
yaw_command_progress.weight: 0.20 -> 1.00
yaw_wheel_differential_progress.weight: 0 -> 1.50
yaw_stuck_with_command.weight: -5.0 -> -7.0
```

含义：让 yaw 命令下“真实身体角速度”和“轮差速”成为更强目标。

#### 5. yaw 时加强 hipx 姿态约束

普通 flat：

```python
yaw_turn_joint_posture_l2.weight: -0.25 -> -1.20
joint_deviation_hipx_l1.weight: -0.15 -> -0.22
```

Yaw 专项 flat：

```python
yaw_turn_joint_posture_l2.weight: -0.25 -> -1.40
```

含义：yaw 时更明确地告诉策略不要靠 hipx 大幅摆动/拧腿来转。

#### 6. yaw 时降低脚滑惩罚

普通 flat：

```python
feet_slide.weight: -1.00 -> -0.85
yaw_slide_scale = 0.25
```

Yaw 专项 flat：

```python
feet_slide.weight: -1.20 -> -0.70
yaw_slide_scale = 0.20
```

含义：保留基础防滑，但不要让原地 yaw 被脚滑惩罚彻底压住。

### 没有修改什么

- 没有修改 `jk03.py` 中任何 JK03 初始物理参数。
- 没有修改 JK03 URDF。
- 没有修改 fan-ziqi 原始 terrain curriculum 算法。
- 没有修改 rough 的 terrain level 计算逻辑。
- 没有恢复强制抬腿/对角踏步奖励，当前目标仍是先把 flat 的基础 yaw 转起来。

### 验证结果

- 本地 `python3 -m py_compile` 通过：
  - `rewards.py`
  - `rough_env_cfg.py`
  - `flat_env_cfg.py`
- 受保护文件 diff 为空。

### 已知风险

- 如果轮子关节速度正负方向和预期不同，新 reward 仍然不会直接写死正负方向，而是用“轮速差值 + 身体真实 yaw 方向”避免符号写反；但训练初期可能需要重新探索轮差速动作。
- 由于限制了腿部 action clip，flat 上会更难出现夸张扭腿，但如果后续 rough 需要大幅抬腿，不能直接把 flat 的 clip 原样套到 rough。
- 这一版目标是解决“拧 45 度卡住”，不是一步到位解决楼梯抬腿。

### 下一步观察指标

- `track_ang_vel_z_exp` 是否上升。
- `yaw_wheel_differential_progress` 是否从 0 变为稳定正值。
- `yaw_stuck_with_command` 绝对值是否下降。
- `yaw_turn_joint_posture_l2` 和 `joint_deviation_hipx_l1` 是否下降。
- `feet_slide` 是否没有因为 yaw 放开而突然发散。
- play 测试里 `base_wz` 是否能持续跟随 Z/X，而不是只扭一下腿。

## 2026-06-18: flat-basic-motion-v5

状态：本地验证通过，已同步云端并通过云端编译，随本次提交推送 GitHub。

### 为什么修改

用户实测反馈：最新版普通 flat 训练出来后“几乎一点都动不了”。结合云端 TensorBoard 趋势：

- `mean_reward` 和 `mean_episode_length` 在上升，说明它学会了更稳定地活着。
- 但 `yaw_turn_tangential_swing`、`yaw_turn_feet_clearance` 仍然很低，说明抬腿转向并没有真正学出来。
- `feet_slide`、`joint_deviation_hipx_l1` 等项仍在拉扯策略。
- 普通 `Flat-JK03-v0` 同时承担“基础移动 + yaw 抬腿 + 对角踏步 + 防滑 + 姿态约束”，奖励目标太多，早期策略容易选择少动或不动。

本次方向改为做减法：

- 普通 flat 阶段先不训练前后脚抬起、对角踏步和 yaw 专项摆腿。
- 先让狗学会前进、后退、左右平移和基本 yaw 命令下能动。
- 防滑只保留为脚滑惩罚，不再用一堆抬腿函数硬逼动作。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 普通 `JK03FlatEnvCfg` 里关闭 yaw 抬腿/对角踏步奖励

将以下普通 flat 权重改为 `0`：

- `feet_gait.weight = 0`
- `yaw_turn_feet_clearance.weight = 0`
- `yaw_turn_diagonal_step.weight = 0`
- `yaw_turn_air_time_deficit.weight = 0`
- `yaw_turn_phase_timeout.weight = 0`
- `yaw_turn_tangential_swing.weight = 0`

含义：普通 flat 不再奖励“抬脚”“对角踏步”“切向摆腿”。这些目标会让还不会稳定运动的早期策略害怕动作，甚至选择站住。

#### 2. 保留并增强脚滑惩罚

将：

```python
feet_slide.weight = -0.70
```

改为：

```python
feet_slide.weight = -1.00
```

含义：当前阶段不是强迫抬腿，而是避免它靠脚/轮子乱滑来骗速度。

#### 3. 提高基础速度跟踪

将：

```python
track_lin_vel_xy_exp.weight = 8.0
track_ang_vel_z_exp.weight = 1.6
```

改为：

```python
track_lin_vel_xy_exp.weight = 9.0
track_ang_vel_z_exp.weight = 1.8
```

含义：让“按键后真的动起来”重新成为普通 flat 的主目标。

#### 4. 放松关节姿态惩罚

将：

```python
commanded_joint_posture_l2.weight = -0.8
front_joint_posture_l2.weight = -0.70
yaw_turn_joint_posture_l2.weight = -1.00
joint_deviation_hipx_l1.weight = -0.35
joint_pos_penalty.weight = -0.55
```

改为：

```python
commanded_joint_posture_l2.weight = -0.45
front_joint_posture_l2.weight = -0.35
yaw_turn_joint_posture_l2.weight = -0.25
joint_deviation_hipx_l1.weight = -0.15
joint_pos_penalty.weight = -0.35
```

含义：之前这些惩罚叠加后，会让策略觉得动腿、动 hipx 都很危险。现在先允许它动起来，后续再逐步收姿态。

#### 5. `JK03FlatYawEnvCfg` 也临时关闭抬腿专项项

将 yaw 专项里的：

- `feet_gait`
- `yaw_turn_feet_clearance`
- `yaw_turn_diagonal_step`
- `yaw_turn_air_time_deficit`
- `yaw_turn_phase_timeout`
- `yaw_turn_tangential_swing`

权重也改为 `0`，避免误跑 yaw 专项时又进入“奖励太复杂导致不动”的问题。保留：

```python
feet_slide.weight = -1.20
track_ang_vel_z_exp.weight = 1.2
```

含义：yaw 专项也先回到“能转、少滑”，不再追求抬腿转向。

### 没有修改什么

- 没有修改 `jk03.py`。
- 没有修改 JK03 URDF。
- 没有修改 fan-ziqi terrain curriculum。
- 没有修改 `curriculums.py`。
- 没有删除之前新增的 reward 函数，只是把普通 flat/yaw 中相关权重关掉。

### 已知风险

- 这一版不再追求“抬腿转向”，所以视觉上可能更像轮足滑/滚动转向。
- 只用脚滑惩罚不一定能让动作像宇树机械狗，但能先排除“不动”的问题。
- 如果 `feet_slide=-1.00` 仍然过强，下一步应降到 `-0.6~-0.8`，而不是重新加抬腿函数。

### 下一步观察

优先观察：

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Metrics/base_velocity/error_vel_xy`
- `Metrics/base_velocity/error_vel_yaw`
- `Episode_Reward/feet_slide`
- `Episode_Reward/joint_pos_penalty`
- `Episode_Reward/joint_deviation_hipx_l1`

用户测试重点：

- 上下键能否前进/后退。
- 左右键能否左右平移。
- Z/X 能否 yaw 转向。
- 身体是否还会直接蹲住不动。

## 2026-06-17: yaw-step-cycle-v4

状态：本地验证通过，已同步云端并通过云端编译，随本次提交推送 GitHub。

### 为什么修改

用户实测反馈：仍不能稳定抬腿转向，转到约 45 度时关节会卡住。

结合上一次 TensorBoard 到约 `step 1418` 的趋势：

- `yaw_turn_feet_clearance` 已经到 `0.02` 左右，说明上一版确实让它开始产生抬轮。
- `yaw_turn_air_time_deficit` 持续改善，说明“不抬腿”的情况减少。
- 但 `yaw_turn_diagonal_step`、`feet_gait` 下降，`feet_slide` 持续变差。
- 这说明策略学到的是“抬一点 + 滑一点”，不是干净的对角交替踏步。
- 原来的 `yaw_turn_diagonal_step` / `yaw_turn_air_time_deficit` 用 `max(pair_0_phase, pair_1_phase)`，只要一组对角脚持续离地、另一组持续支撑，就可能得分。这会鼓励“单边对角相位卡住”，对应用户看到的 45 度附近关节卡死。

本次修改目标：

- 不再只奖励“有一组对角脚离地”。
- 增加最大离地/最大支撑时间，超过时间后不给分或扣分，逼它换脚。
- 增加 yaw 转向时空中轮子的切向摆动奖励，让它抬起来以后必须朝正确转向方向摆，而不是只抬一下。
- 增强抗滑约束，减少“抬一点 + 滑一点”的投机策略。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 新增 `_bounded_phase_score`

新增内部工具函数：

```python
_bounded_phase_score(time, min_time, max_time)
```

逻辑：

- 小于 `min_time`：分数从 0 增长到 1。
- 位于合理窗口：分数高。
- 超过 `max_time`：分数下降到 0。

目的：防止一组对角脚长时间离地或另一组长时间支撑，从而卡在一个相位不换脚。

#### 2. 修改 `yaw_turn_diagonal_step`

新增参数：

```python
max_air_time
max_contact_time
```

原来只看：

```python
air_time / min_air_time
contact_time / min_contact_time
```

现在改成：

```python
_bounded_phase_score(air_time, min_air_time, max_air_time)
_bounded_phase_score(contact_time, min_contact_time, max_contact_time)
```

含义：一组对角脚可以离地，但不能一直离地；另一组可以支撑，但不能一直支撑。这样才会逼出周期性交替。

#### 3. 修改 `yaw_turn_air_time_deficit`

同样新增：

```python
max_air_time
max_contact_time
```

原来只要存在一组对角离地/另一组支撑，惩罚就会降低。现在如果这个相位保持太久，得分会掉下来，惩罚会重新变大。

#### 4. 新增 `yaw_turn_phase_timeout`

新增惩罚函数：

```python
yaw_turn_phase_timeout(...)
```

算法：

```python
air_timeout = relu(air_time - max_air_time) / max_air_time
contact_timeout = relu(contact_time - max_contact_time) / max_contact_time
penalty = mean(air_timeout^2 + contact_timeout^2)
```

含义：yaw 命令下，如果某些轮子离地太久或支撑太久，就直接扣分。这个项是专门针对“转到 45 度卡住不换脚”的。

#### 5. 新增 `yaw_turn_tangential_swing`

新增奖励函数：

```python
yaw_turn_tangential_swing(...)
```

算法核心：

```python
tangent = [-foot_y, foot_x]
signed_tangent_speed = dot(foot_velocity_xy, tangent_dir) * sign(yaw_command)
reward = clearance * air_score * positive_tangential_speed * opposite_pair_contact
```

含义：yaw 转向时，空中的轮子不只是抬起来，还要沿着绕机身旋转的切向方向摆。这样奖励从“抬脚”升级为“抬脚并向正确转向方向迈步”。

#### 6. Flat 权重调整

调整：

```python
feet_slide.weight: -0.50 -> -0.70
feet_gait.weight: 1.80 -> 2.20
yaw_turn_feet_clearance.weight: 7.50 -> 6.50
yaw_turn_feet_clearance.max_air_time: 0.32 -> 0.20
yaw_turn_diagonal_step.weight: 5.50 -> 7.50
yaw_turn_diagonal_step.max_air_time = 0.18
yaw_turn_diagonal_step.max_contact_time = 0.24
yaw_turn_air_time_deficit.max_air_time = 0.18
yaw_turn_air_time_deficit.max_contact_time = 0.24
yaw_turn_phase_timeout.weight = -2.50
yaw_turn_tangential_swing.weight = 4.00
yaw_turn_joint_posture_l2.weight: -0.75 -> -1.00
```

关键意图：

- 不再继续单纯加大抬轮高度奖励，避免“只抬不走”。
- 加强对角步态和切向摆腿。
- 加强抗滑。
- 加强 yaw 时 hipx 姿态约束，减少关节扭到卡住。

#### 7. Flat-Yaw 专训权重调整

调整：

```python
feet_slide.weight: -0.65 -> -0.90
feet_gait.weight: 2.50 -> 3.00
yaw_turn_feet_clearance.weight: 10.00 -> 8.50
yaw_turn_diagonal_step.weight: 8.00 -> 10.00
yaw_turn_air_time_deficit.weight: -5.00 -> -6.00
yaw_turn_phase_timeout.weight = -4.00
yaw_turn_tangential_swing.weight = 6.00
yaw_turn_joint_posture_l2.weight: -0.55 -> -1.00
```

### 没有修改什么

- 没有修改 `jk03.py`。
- 没有修改 JK03 URDF。
- 没有修改 fan-ziqi 原始 terrain curriculum。
- 没有修改 terrain level 算法。

### 已知风险

- 这版会更强地限制“吊脚不换”和“滑着转”，训练早期 reward 可能下降。
- 如果约束太强，策略可能短时间更保守，需要重新从头训练观察。
- 旧 checkpoint 不能直接代表新 reward 效果，建议新开 run。

### 下一步观察指标

- `yaw_turn_phase_timeout`：应该逐步接近 0。
- `yaw_turn_tangential_swing`：应该逐步上升。
- `yaw_turn_diagonal_step`：应该回升并稳定高于 `0.14`。
- `feet_slide`：不能继续低于 `-0.70`。
- 视频中应该看到对角脚轮交替抬起，而不是一组对角脚一直吊着。

## 2026-06-17: flat-yaw-lift-aggressive-v3

状态：本地验证通过，已同步云端并通过云端编译，已提交并推送 GitHub。

### 为什么修改

Flat 训练到约 `step 1930` 后，趋势显示：

- `mean_reward`、`xy velocity error`、`yaw velocity error` 继续改善，说明训练整体没有发散。
- `joint_deviation_hipx_l1` 和 `joint_pos_penalty` 变好，说明关节强扭和整体姿态更稳定。
- 但 `yaw_turn_feet_clearance` 从 `0.00754 -> 0.00657` 下降。
- `yaw_turn_diagonal_step` 从 `0.0854 -> 0.0740` 下降。
- `feet_gait` 从 `0.0998 -> 0.0839` 下降。

这说明策略学会了更稳地跟速度和保持姿态，但仍在回避“抬轮/对角踏步转向”，可能继续靠贴地小滑动或低幅关节动作完成 yaw。

本次修改目标是更激进地让它先学会：有原地 yaw 命令时，必须出现一组对角轮离地、另一组对角轮支撑的相位。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 新增 `yaw_turn_air_time_deficit`

新增 reward 函数：

```python
yaw_turn_air_time_deficit(...)
```

它只在 yaw-only 命令下生效：

```python
abs(command_yaw) > command_threshold
norm(command_xy) <= max_xy_command
```

算法核心：

```python
pair_0_swing_support = pair_0_air * pair_1_contact
pair_1_swing_support = pair_1_air * pair_0_contact
best_diagonal_phase = max(pair_0_swing_support, pair_1_swing_support)
penalty = square(1.0 - best_diagonal_phase)
```

含义：

- `pair_0_air`：第一组对角轮 `fl_wheel + hr_wheel` 的离地时间得分。
- `pair_1_air`：第二组对角轮 `fr_wheel + hl_wheel` 的离地时间得分。
- `pair_0_contact` / `pair_1_contact`：对应对角组的支撑接触时间得分。
- 如果一组对角轮离地、另一组对角轮支撑，`best_diagonal_phase` 接近 1，惩罚接近 0。
- 如果 yaw 命令下四个轮都贴地，没有抬轮相位，惩罚接近 1。

这个项不是奖励“转得快”，而是直接惩罚“不抬腿转向”。

#### 2. 主 Flat 更偏向抬轮转向

降低容易产生贴地转向捷径的项：

```python
track_ang_vel_z_exp.weight: 2.4 -> 1.6
yaw_command_progress.weight: 0.8 -> 0.35
```

放松普通关节和动作变化约束，让 hipy/knee 有空间抬轮：

```python
joint_pos_penalty.weight: -0.75 -> -0.55
action_rate_l2.weight: -0.003 -> -0.002
```

但仍保留 hipx 约束，避免靠横向强拧转向：

```python
yaw_turn_joint_posture_l2.weight: -1.0 -> -0.75
```

大幅加强抬轮和对角踏步：

```python
feet_slide.weight: -0.34 -> -0.50
feet_gait.weight: 1.20 -> 1.80
feet_gait.std: 0.55 -> 0.45
yaw_turn_feet_clearance.weight: 4.00 -> 7.50
yaw_turn_feet_clearance.target_height: -0.245 -> -0.235
yaw_turn_feet_clearance.min_air_time: 0.015 -> 0.010
yaw_turn_feet_clearance.max_air_time: 0.28 -> 0.32
yaw_turn_feet_clearance.tanh_mult: 6.0 -> 7.5
yaw_turn_diagonal_step.weight: 3.00 -> 5.50
yaw_turn_diagonal_step.phase_balance_weight: 0.05 -> 0.0
yaw_turn_air_time_deficit.weight: 0.0 -> -3.00
```

#### 3. Flat-Yaw 专训更激进

Flat-Yaw 任务继续只训练原地 yaw：

```python
lin_vel_x = (0.0, 0.0)
lin_vel_y = (0.0, 0.0)
ang_vel_z = (-0.9, 0.9)
```

这次让它更专注于抬轮，而不是先追 yaw 速度：

```python
track_ang_vel_z_exp.weight = 1.0
yaw_command_progress.weight = 0.20
yaw_stuck_with_command.weight = -5.0
feet_slide.weight = -0.65
feet_gait.weight = 2.50
yaw_turn_feet_clearance.weight = 10.00
yaw_turn_diagonal_step.weight = 8.00
yaw_turn_air_time_deficit.weight = -5.00
joint_pos_penalty.weight = -0.35
yaw_turn_joint_posture_l2.weight = -0.55
```

### 没有修改什么

- 没有修改 `jk03.py`。
- 没有修改 JK03 URDF。
- 没有修改 fan-ziqi 原始 terrain curriculum。
- 没有修改 terrain level 算法。

### 推荐训练顺序

如果当前 Flat 到 2000 步仍不会抬轮，建议不要继续只用普通 Flat 硬跑，先切到 yaw-only 预训练：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK03-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 3000
```

观察：

- `yaw_turn_air_time_deficit` 应明显下降。
- `yaw_turn_feet_clearance` 应稳定升到 `0.01+`。
- `yaw_turn_diagonal_step` 应稳定升到 `0.10+`。
- `feet_slide` 不应继续变得更负。

## 2026-06-16: flat-yaw-lift-pretrain-v2

状态：本地验证通过，已同步云端并通过云端编译，已提交并推送 GitHub。

### 为什么修改

最新 Flat 训练到 `step 1449` 时，TensorBoard 显示：

- `mean_reward`、`xy error`、`yaw error` 总体在改善，说明训练没有发散。
- `yaw_turn_diagonal_step` 有小幅上升，但绝对值仍低。
- `yaw_turn_feet_clearance` 从 `0.00317 -> 0.00292 -> 0.00281`，说明抬轮没有形成，甚至变弱。
- `feet_slide` 仍偏大，策略仍可能靠贴地滑动、小幅关节扭动来完成 yaw。

所以这次不继续只做小幅权重微调，而是按“先把原地 yaw 抬轮/对角踏步单独学出来”的思路修改：

- 把 yaw 抬轮 reward 从硬阈值改成连续奖励，让早期小抬轮也能得到学习信号。
- 进一步降低纯 yaw 速度奖励，减少贴地滑动投机。
- 加强对角踏步、抬轮和滑动惩罚。
- 新增一个 Flat-Yaw 专训任务，只训练原地 yaw，不训练前后/左右平移。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/__init__.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/agents/rsl_rl_ppo_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/agents/cusrl_ppo_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. `yaw_turn_feet_clearance` 改成连续 air/contact 信号

原来是硬判断：

```python
airborne = (air_time > min_air_time).float()
pair_ground = (contact_time > min_contact_time).float()
```

问题：训练早期只要还没抬够 `min_air_time`，奖励就是 0，PPO 很难知道“稍微抬一点是对的”。

现在改成：

```python
air_score = clamp(air_time / min_air_time, 0, 1)
contact_score = clamp(contact_time / min_contact_time, 0, 1)
```

含义：还没完全达到阈值时也有 0 到 1 的渐进奖励，小抬轮、小接触保持都会产生学习方向。

#### 2. `yaw_turn_diagonal_step` 也改成连续相位

原来：

```python
pair_air = mean(air_time > min_air_time)
pair_contact = mean(contact_time > min_contact_time)
```

现在：

```python
pair_air = mean(clamp(air_time / min_air_time, 0, 1))
pair_contact = mean(clamp(contact_time / min_contact_time, 0, 1))
```

这样 `fl+hr` 与 `fr+hl` 的对角摆动/支撑关系更早能被奖励到，不用等完全超过硬阈值。

#### 3. 主 Flat 降低纯 yaw 速度奖励

修改：

```python
track_ang_vel_z_exp.weight: 3.2 -> 2.4
yaw_command_progress.weight: 1.2 -> 0.8
```

原因：如果 yaw 速度奖励太高，策略会优先找“贴地滑动、扭关节也能转起来”的捷径。

#### 4. 主 Flat 加强抬轮/对角踏步/滑动约束

修改：

```python
feet_slide.weight: -0.26 -> -0.34
feet_gait.weight: 0.90 -> 1.20
yaw_turn_feet_clearance.weight: 2.40 -> 4.00
yaw_turn_feet_clearance.target_height: -0.255 -> -0.245
yaw_turn_feet_clearance.min_air_time: 0.025 -> 0.015
yaw_turn_feet_clearance.max_air_time: 0.25 -> 0.28
yaw_turn_feet_clearance.tanh_mult: 5.0 -> 6.0
yaw_turn_diagonal_step.weight: 1.80 -> 3.00
yaw_turn_diagonal_step.min_air_time: 0.025 -> 0.015
yaw_turn_diagonal_step.phase_balance_weight: 0.20 -> 0.05
```

含义：

- 更强惩罚贴地滑动。
- 更强奖励对角步态。
- 抬轮目标更高一点。
- `min_air_time` 降低，是为了让早期探索更容易拿到连续奖励，不是降低最终要求。
- `phase_balance_weight` 降低，是为了减少“四轮都差不多状态”时的保底分，更偏向一组对角脚摆动、另一组支撑。

#### 5. 减轻普通关节惩罚，避免压住 hipy/knee 抬腿

修改：

```python
joint_pos_penalty.weight: -1.1 -> -0.75
```

原因：普通关节惩罚太强时，策略可能不敢弯 `hipy/knee` 抬腿。保留 `yaw_turn_joint_posture_l2` 继续限制 `hipx` 横向强拧，但放松一般关节变化，让抬腿有空间。

#### 6. 新增 Flat-Yaw 专训任务

新增任务：

```text
RobotLab-Isaac-Velocity-Flat-Yaw-JK03-v0
```

对应环境：

```python
JK03FlatYawEnvCfg
```

命令范围：

```python
lin_vel_x = (0.0, 0.0)
lin_vel_y = (0.0, 0.0)
ang_vel_z = (-0.9, 0.9)
```

奖励更集中：

```python
track_ang_vel_z_exp.weight = 1.6
yaw_command_progress.weight = 0.45
yaw_stuck_with_command.weight = -4.0
feet_slide.weight = -0.45
feet_gait.weight = 1.60
yaw_turn_feet_clearance.weight = 5.50
yaw_turn_diagonal_step.weight = 4.00
joint_pos_penalty.weight = -0.55
yaw_turn_joint_posture_l2.weight = -0.80
```

原因：先单独训练“原地 yaw 时必须抬轮/对角踏步”，不要让 PPO 同时学习前进、后退、左右平移，降低探索难度。

### 没有修改什么

- 没有修改 `jk03.py`。
- 没有修改 JK03 URDF。
- 没有修改 fan-ziqi 原始 terrain curriculum。
- 没有修改 rough 的 terrain level 算法。

### 推荐训练顺序

先专训 yaw：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Yaw-JK03-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 3000
```

如果 `yaw_turn_feet_clearance` 和 `yaw_turn_diagonal_step` 明显上升，再回到完整 Flat：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 5000
```

### 下一步观察指标

- `yaw_turn_feet_clearance`：希望从 `0.002-0.005` 提升到 `0.01+`。
- `yaw_turn_diagonal_step`：希望从 `0.02-0.04` 提升到 `0.06+`。
- `feet_slide`：不能继续变得更负。
- `yaw error`：应缓慢下降。
- 视频中应看到对角轮组交替离地，而不是身体趴下贴地滑动。

## 2026-06-16: flat-lateral-keyboard-diagonal-step-v1

状态：本地验证通过，已同步云端，已提交并推送 GitHub。

### 为什么修改

用户实测发现两个问题：

- 左右键语义不对：用户期望左右箭头是左右平移 `lin_vel_y`，而不是原地 yaw 转向；`Z/X` 才应该控制头部/机身 yaw 转向。
- 最新 Flat 训练到约 1800 step 后仍然不能稳定抬轮/对角踏步转向。TensorBoard 也显示：
  - `yaw_turn_feet_clearance` 仍低于 `0.01`，并且最近窗口下降。
  - `feet_gait` 没有稳定上升。
  - `feet_slide` 偏大，说明策略仍偏向贴地滑动。

上一版主要靠 `yaw_turn_feet_clearance`，这个信号太稀疏，策略没有明显学到“一个对角组离地、另一个对角组支撑”。所以这次修改幅度更大：修正键盘语义、打开左右平移训练，并新增一个更直接的对角踏步接触相位 reward。

### 修改文件

- `robot_lab-main/scripts/reinforcement_learning/rsl_rl/play.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 修改键盘控制语义

在 `play.py` 中删除“当 `lin_vel_y=0` 时把左右箭头映射成 yaw”的逻辑，改为固定映射：

```text
上箭头 / W        -> vx 正向，前进
下箭头 / S        -> vx 负向，后退
左箭头 / A        -> vy 正向，向左平移
右箭头 / D        -> vy 负向，向右平移
Z / Q             -> yaw 正向
X / E             -> yaw 负向
```

同时把键盘 yaw 灵敏度从：

```python
0.45 * ang_vel_z_max
```

提高到：

```python
0.70 * ang_vel_z_max
```

原因：`Z/X` 明确负责 yaw 后，可以给更明显的 yaw 输入；左右键不再被拿来转向。

#### 2. 打开 Flat 左右平移训练

在 `flat_env_cfg.py` 中修改：

```python
lin_vel_y: (0.0, 0.0) -> (-0.45, 0.45)
```

原因：如果训练时 `lin_vel_y` 一直是 0，那么键盘左右平移即使能发出 `vy`，策略也没学过这个动作。打开后，Flat 预训练会同时学习前后、左右平移和 yaw。

#### 3. 新增 `yaw_turn_diagonal_step`

在 `rewards.py` 中新增 reward 函数：

```python
yaw_turn_diagonal_step
```

它只在接近原地 yaw 转向时启用：

```python
abs(yaw_command) > command_threshold
norm(command_xy) <= max_xy_command
```

核心逻辑：

```python
pair_0 = ("fl_wheel", "hr_wheel")
pair_1 = ("fr_wheel", "hl_wheel")

pair_0_air = mean(pair_0 air_time > min_air_time)
pair_1_air = mean(pair_1 air_time > min_air_time)
pair_0_contact = mean(pair_0 contact_time > min_contact_time)
pair_1_contact = mean(pair_1 contact_time > min_contact_time)

pair_0_swing = pair_0_air * pair_1_contact
pair_1_swing = pair_1_air * pair_0_contact
reward = max(pair_0_swing, pair_1_swing)
```

含义：

- `fl+hr` 离地、`fr+hl` 支撑，给分。
- `fr+hl` 离地、`fl+hr` 支撑，给分。
- 四轮贴地滑动，不给这个奖励。
- 四轮同时乱跳，也不会拿到高分。

这个 reward 比 `yaw_turn_feet_clearance` 更直接，先告诉策略“转向时应该出现对角支撑/摆动相位”；`yaw_turn_feet_clearance` 再负责让摆动脚/轮真的抬高。

#### 4. Flat 中启用并加强对角踏步/抬轮

新增启用：

```python
yaw_turn_diagonal_step.weight = 1.80
```

加强原有项：

```python
feet_gait.weight: 0.45 -> 0.90
feet_gait.std: sqrt(0.5) -> 0.55
feet_gait.max_err: 0.25 -> 0.30
yaw_turn_feet_clearance.weight: 1.25 -> 2.40
yaw_turn_feet_clearance.target_height: -0.27 -> -0.255
yaw_turn_feet_clearance.min_air_time: 0.02 -> 0.025
yaw_turn_feet_clearance.max_air_time: 0.18 -> 0.25
yaw_turn_feet_clearance.tanh_mult: 4.0 -> 5.0
yaw_turn_feet_clearance.min_base_height: 0.43 -> 0.435
yaw_turn_feet_clearance.diagonal_pair_weight: 0.90 -> 0.95
```

原因：

- `feet_gait` 负责对角时序。
- `yaw_turn_diagonal_step` 负责“一个对角组离地、另一个对角组支撑”。
- `yaw_turn_feet_clearance` 负责真的抬起来，而不是贴地擦。
- `target_height=-0.255` 比上一版要求更高，逼它形成更明显的抬轮。

#### 5. 降低纯 yaw 速度奖励，减少贴地滑动投机

修改：

```python
track_ang_vel_z_exp.weight: 4.5 -> 3.2
yaw_command_progress.weight: 2.4 -> 1.2
```

原因：之前 yaw 速度奖励比较强，策略可以通过贴地滑动、扭关节来拿 yaw 速度分。现在降低这两项，让“怎么转”比“只要转起来”更重要。

#### 6. 加强滑动和 yaw 姿态约束

修改：

```python
feet_slide.weight: -0.18 -> -0.26
yaw_turn_joint_posture_l2.weight: -0.75 -> -1.0
```

原因：

- `feet_slide` 更强，减少贴地拖着转。
- `yaw_turn_joint_posture_l2` 更强，减少靠 hipx 横向强拧实现转向。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改 PPO。
- 没改 base 初始高度。

### 验证

- 本地 Python 编译检查通过：
  - `play.py`
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
- 云端确认 `lin_vel_y = (-0.45, 0.45)`。
- 云端确认 `yaw_turn_diagonal_step.weight = 1.80`。
- 云端确认 `play.py` 中左/右箭头为 `vy`，`Z/X` 为 `yaw`。
- 云端确认 `velocity_env_cfg.py` 仍为 `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`。
- 云端确认 JK03 rough 没有 `terrain_levels` 覆盖：`NO_JK03_TERRAIN_OVERRIDE`。

### 已知风险

- 这次修改幅度较大，旧 checkpoint 不能代表新策略表现，必须重新开新 Flat 训练。
- `lin_vel_y` 打开后，前期 mean_reward 可能短暂下降，因为任务从前后/yaw 变成前后/左右/yaw。
- 抬轮和滑动惩罚更强，前 500-1000 step 可能看起来动作更犹豫。
- 如果 `yaw_turn_diagonal_step` 上升但 `yaw_turn_feet_clearance` 不上升，说明有对角相位但抬得不够，需要继续提高 clearance 或检查 wheel air-time。

### 下一步观察指标

- `Episode_Reward/yaw_turn_diagonal_step`
- `Episode_Reward/yaw_turn_feet_clearance`
- `Episode_Reward/feet_gait`
- `Episode_Reward/feet_slide`
- `Metrics/base_velocity/error_vel_xy`
- `Metrics/base_velocity/error_vel_yaw`
- 键盘测试时：
  - 左右箭头应输出 `vy`。
  - `Z/X` 应输出 `yaw`。
  - 左右平移需要用新训练的 checkpoint 测，旧 checkpoint 没学过 `lin_vel_y`。

## 2026-06-16: flat-yaw-diagonal-clearance-v2

状态：本地验证通过，已同步云端，已提交并推送 GitHub。

### 为什么修改

Flat 训练到约 1600 step 后，用户实测仍然没有形成“对角踏步原地转向”。上一版 `yaw_turn_feet_clearance` 的 TensorBoard 均值只在 `0.009` 左右平台，说明策略偶尔会抬轮，但没有稳定把“抬起对角轮、另一对角支撑”学成主要转向方式。视频/实测表现仍然偏向贴地滑动、拧关节和转向困难。

这一版不再继续堆很多新 reward，而是把已有 `yaw_turn_feet_clearance` 改得更明确：转向时只有对角组真的离地、另一对角组在地面支撑，奖励才会明显变高。

### 修改文件

- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/rough_env_cfg.py`
- `robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/flat_env_cfg.py`
- `JK03_CHANGELOG.md`

### 怎么修改

#### 1. 重写 `yaw_turn_feet_clearance` 的核心算法

新增可选参数：

```python
synced_feet_pair_names
min_contact_time
diagonal_pair_weight
```

当 `synced_feet_pair_names` 存在时，函数不再只对四个轮子的 clearance 做简单平均，而是按两组对角轮计算：

```python
("fl_wheel", "hr_wheel")
("fr_wheel", "hl_wheel")
```

核心逻辑变成：

```python
pair_0_lift = mean(lift_score of fl/hr)
pair_1_lift = mean(lift_score of fr/hl)
pair_0_ground = mean(contact_time of fl/hr > min_contact_time)
pair_1_ground = mean(contact_time of fr/hl > min_contact_time)

pair_0_swing_phase = pair_0_lift * pair_1_ground
pair_1_swing_phase = pair_1_lift * pair_0_ground
diagonal_reward = max(pair_0_swing_phase, pair_1_swing_phase)
```

含义：

- `fl+hr` 抬起来、`fr+hl` 支撑，给高分。
- `fr+hl` 抬起来、`fl+hr` 支撑，也给高分。
- 四个轮子贴地滑动，几乎不给这个奖励。
- 单个轮子乱抬，但另一对角组没有稳定支撑，奖励也不会高。

最终奖励：

```python
reward = diagonal_pair_weight * diagonal_reward + (1 - diagonal_pair_weight) * average_lift_reward
```

`diagonal_pair_weight=0.90` 表示 90% 的奖励来自对角相位，10% 保留给早期探索，避免训练一开始完全没有梯度。

#### 2. Flat 中加强对角踏步信号

修改：

```python
feet_gait.weight: 0.35 -> 0.45
yaw_turn_feet_clearance.weight: 0.80 -> 1.25
```

原因：

- `feet_gait` 负责对角接触节奏。
- `yaw_turn_feet_clearance` 负责真的抬起来。
- 上一版两者都有改善，但抬轮 reward 平台太低，所以这次把二者同时加强，但没有改 yaw tracking 主奖励，避免策略只追求更快旋转又回到拧关节。

#### 3. Flat 中提高抬轮目标

修改：

```python
target_height: -0.29 -> -0.27
min_air_time: 0.015 -> 0.02
max_air_time: 0.20 -> 0.18
tanh_mult: 3.0 -> 4.0
min_base_height: 0.425 -> 0.43
base_height_margin: 0.025 -> 0.03
```

原因：

- `target_height=-0.27` 比上一版要求更高，防止只是轻微离地。
- `min_air_time=0.02` 防止接触传感器抖一下就算抬腿。
- `max_air_time=0.18` 防止长时间悬空乱跳。
- `tanh_mult=4.0` 让轮/脚水平摆动更快进入奖励区间，鼓励真正换步。
- `min_base_height=0.43` 防止趴下以后用很小动作骗 clearance。

#### 4. 稍微加强滑动惩罚

修改：

```python
feet_slide.weight: -0.16 -> -0.18
```

原因：

上一版 `feet_slide` 仍然略差，说明它还在用贴地滑动转向。这里只小幅加强，避免把前进动作也惩罚得太死。

### 没有修改

- 没改 URDF。
- 没改 `jk03.py` 初始物理参数。
- 没改 fan-ziqi 原始 `terrain_levels_vel`。
- 没在 JK03 rough 配置里覆盖 terrain level 算法。
- 没改 PPO。
- 没改动作空间。
- 没改 base 初始高度。

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

### 已知风险

- 如果 `yaw_turn_feet_clearance` 权重过大，可能出现原地转向小跳步。
- 如果 `feet_slide` 惩罚和 clearance 奖励冲突，早期 yaw 速度可能短暂下降。
- 如果 2000 step 后 `yaw_turn_feet_clearance` 仍然低于 `0.02`，说明接触/抬轮信号仍不够，需要继续查视频和 wheel body 的 air-time 读数。

### 下一步观察指标

- `Episode_Reward/yaw_turn_feet_clearance` 是否从 `0.009` 平台稳定突破 `0.02`。
- `Episode_Reward/feet_gait` 是否继续上升，而不是单点冲高。
- `Episode_Reward/feet_slide` 是否下降或至少不继续恶化。
- `Metrics/base_velocity/error_vel_yaw` 是否继续下降。
- 视频里是否出现 `fl+hr` 与 `fr+hl` 对角组交替抬起。

## 2026-06-16: flat-yaw-air-clearance-v1

状态：本地验证通过，已同步云端，已提交并推送 GitHub。

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
