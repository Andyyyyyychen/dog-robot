# dog_robot

JK03 轮足机器狗仿真与强化学习训练仓库。

本仓库主要包含三部分：

- `robot_lab-main`：Isaac Lab / Isaac Sim 里的 JK03 强化学习训练任务。
- `rl_sar-main`：本地 MuJoCo 手动仿真入口，用来先看 JK03 模型、关节、轮子和场景。
- `robot_lab-main/scripts/tools/check_jk03_pretrain.py`：训练前检查脚本，用来核对 JK03 文件、URDF、网格、Gym 环境注册和 MuJoCo 参数镜像。

注意：不要直接修改 JK03 初始数据和参数。受保护路径包括：

```text
robot_lab-main/source/robot_lab/robot_lab/assets/jk03.py
robot_lab-main/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/jk03/
robot_lab-main/source/robot_lab/data/Robots/jk03/
```

## 1. 拉取仓库

推荐用 `git clone`，不要用 zip：

```bash
cd ~
git clone https://github.com/Andyyyyyychen/dog-robot.git dog_robot
cd ~/dog_robot
git status
```

如果云服务器因为防火墙无法访问 GitHub，可以在本地电脑打包后用 `scp` 传上去：

```bash
# 本地电脑
cd /path/to
tar -czf dog_robot.tar.gz dog_robot
scp -P <SSH_PORT> dog_robot.tar.gz <USER>@<SERVER_IP>:~/

# 云服务器
cd ~
tar -xzf dog_robot.tar.gz
cd dog_robot
```

## 2. 本地 MuJoCo 查看 JK03

这个步骤不训练模型，只是在本地 Mac 或 Linux 上查看 JK03 是否能正常显示和手动移动。

安装 MuJoCo：

```bash
python3 -m pip install mujoco
```

进入仓库根目录运行默认楼梯场景：

```bash
cd /path/to/dog_robot
python3 rl_sar-main/scripts/manual_mujoco_jk03.py
```

其他常用场景：

```bash
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene low-stairs
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --collision-only --camera fixed
```

如果出现乱转、漂移、场景像消失或画面很奇怪，先用这个稳定检查命令：

```bash
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground --collision-only --camera fixed --stand
```

MuJoCo 常用按键：

```text
1 开始轮驱    0 站立       W/S 前进/后退
A/D 左右偏置  Q/E 转向      Space 清空指令
R 重置姿态    H 帮助        Esc 退出
```

详细说明见：

```text
rl_sar-main/docs/manual_mujoco_jk03_cn.md
```

## 3. 训练前检查

在仓库根目录运行：

```bash
cd /path/to/dog_robot
python3 robot_lab-main/scripts/tools/check_jk03_pretrain.py
```

如果本机没有 MuJoCo，或者只想做静态检查：

```bash
python3 robot_lab-main/scripts/tools/check_jk03_pretrain.py --skip-mujoco
```

理想结果是：

```text
0 failure(s)
```

如果在 Mac 上看到 Isaac Lab 或 NVIDIA GPU 的 warning，一般没关系，因为正式训练是在 Linux + NVIDIA GPU 云服务器上完成。

## 4. 云服务器环境

推荐云服务器配置：

```text
GPU: RTX 4090 24GB 或更高
系统: Ubuntu 22.04
内存: 至少 32GB，推荐 50GB+
磁盘: 至少 150GB，推荐 200GB+
镜像: 优先选择已经安装 Isaac Sim + Isaac Lab 的镜像
```

登录服务器后先检查显卡：

```bash
nvidia-smi
```

能看到 RTX 4090、Driver Version、显存信息，说明 GPU 基本正常。

如果租用的是预装 Isaac Sim / Isaac Lab 的镜像，通常先激活环境：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
```

当前测试过的 Isaac Lab 启动脚本路径是：

```text
/root/IsaacLab/isaaclab.sh
```

如果你的服务器路径不同，把下面命令里的 `/root/IsaacLab/isaaclab.sh` 换成你自己的 Isaac Lab 启动脚本。

## 5. 安装 robot_lab 工程

在云服务器上执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
cd ~/dog_robot/robot_lab-main
python -m pip install -e source/robot_lab
```

如果你的项目在这个路径：

```text
/root/dog-robot-main/robot_lab-main
```

就执行：

```bash
cd /root/dog-robot-main/robot_lab-main
python -m pip install -e source/robot_lab
```

检查 JK03 环境是否注册成功：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py --keyword JK03
```

必须能看到这两个环境：

```text
RobotLab-Isaac-Velocity-Flat-JK03-v0
RobotLab-Isaac-Velocity-Rough-JK03-v0
```

注意：本仓库里的 JK03 环境名不是 `RobotLab-Isaac-Velocity-Flat-Deeprobotics-JK03-v0`。

## 6. 最小测试流程

训练和测试优先使用 `--headless`，不要一开始就打开 Isaac Sim 图形界面。云服务器 Desktop 图形界面比较卡，适合短时间看效果，不适合长时间训练。

下面的测试命令按云服务器上的 Rough 版本编写：

进入项目：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
cd /root/dog-robot-main/robot_lab-main
```

零动作测试：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/tools/zero_agent.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --headless \
  --num_envs 16
```

随机动作测试：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/tools/random_agent.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --headless \
  --num_envs 16
```

RSL-RL 一轮训练烟测：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --headless \
  --num_envs 16 \
  --max_iterations 1
```

如果没有出现 `Traceback`，说明代码、环境注册、机器人生成和训练入口基本都通了。

## 7. 短训练测试

先跑 100 次，不要一开始就长时间训练：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 100
```

成功时会看到类似：

```text
Learning iteration 99/100
Training time: ...
```

训练结果会写到：

```text
robot_lab-main/logs/rsl_rl/jk03_rough/<run_time>/
```

查找最新模型：

```bash
find logs/rsl_rl/jk03_rough -name "model_*.pt" | sort | tail
```

例如可能看到：

```text
logs/rsl_rl/jk03_rough/2026-06-09_11-05-00/model_99.pt
```

## 8. 正式训练

Flat 平地训练：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 2000
```

如果显存够，可以把 `--num_envs` 改成 `1024`。如果报显存不足，就降到 `512`、`256` 或 `128`。

Flat 稳定后再训练 Rough：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 3000
```

Rough 训练结果会写到：

```text
robot_lab-main/logs/rsl_rl/jk03_rough/<run_time>/
```

建议开另一个终端观察显存：

```bash
watch -n 2 nvidia-smi
```

## 9. 播放训练后的模型

把 `<run_time>` 和 `<N>` 换成实际的 checkpoint 目录和模型编号。

推荐方式：云端 headless 录视频，然后下载到 Mac 看：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --checkpoint logs/rsl_rl/jk03_rough/<run_time>/model_<N>.pt \
  --headless \
  --video \
  --video_length 500 \
  --num_envs 1
```

视频目录：

```text
logs/rsl_rl/jk03_rough/<run_time>/videos/play/
```

如果要在云服务器 Desktop 里打开 Isaac Sim 窗口并用键盘控制，使用：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --checkpoint logs/rsl_rl/jk03_rough/<run_time>/model_<N>.pt \
  --keyboard \
  --num_envs 1 \
  --real-time
```

键盘控制时不要加 `--headless`。Isaac Sim 窗口出现后，先用鼠标点一下仿真视口，让键盘焦点进入窗口。

`--keyboard` 使用 Isaac Lab 的 `Se2Keyboard`，发送的是速度指令。常用的是方向键或小键盘方向键；如果 `WASD` 没反应，先用方向键和数字小键盘测试。

## 10. 在 Mac 上看云端训练结果

### 方法 A：SSH 隧道看 TensorBoard

服务器上运行：

```bash
cd /root/dog-robot-main/robot_lab-main
tensorboard --logdir logs/rsl_rl/jk03_rough --host 127.0.0.1 --port 6006
```

Mac 另开一个终端：

```bash
ssh -p <SSH_PORT> -L 6006:127.0.0.1:6006 <USER>@<SERVER_IP>
```

然后在 Mac 浏览器打开：

```text
http://127.0.0.1:6006
```

如果 Mac 的 `6006` 被占用，可以改成：

```bash
ssh -p <SSH_PORT> -L 6007:127.0.0.1:6006 <USER>@<SERVER_IP>
```

然后打开：

```text
http://127.0.0.1:6007
```

### 方法 B：下载日志到 Mac

```bash
mkdir -p ~/Desktop/jk03_results
scp -P <SSH_PORT> -r <USER>@<SERVER_IP>:/root/dog-robot-main/robot_lab-main/logs/rsl_rl/jk03_rough/<run_time> ~/Desktop/jk03_results/
```

Mac 本地看 TensorBoard：

```bash
python3 -m pip install tensorboard
tensorboard --logdir ~/Desktop/jk03_results --port 6006
open http://127.0.0.1:6006
```

下载视频：

```bash
mkdir -p ~/Desktop/jk03_videos
scp -P <SSH_PORT> -r <USER>@<SERVER_IP>:/root/dog-robot-main/robot_lab-main/logs/rsl_rl/jk03_rough/<run_time>/videos/play ~/Desktop/jk03_videos/
open ~/Desktop/jk03_videos/play
```

## 11. 常见问题

### Isaac Sim 打开一下就没了

通常是脚本结束或报错了。看终端最后有没有：

```text
Traceback
Error
ImportError
FileNotFoundError
```

保存播放日志：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --checkpoint logs/rsl_rl/jk03_rough/<run_time>/model_<N>.pt \
  --keyboard \
  --num_envs 1 \
  --real-time 2>&1 | tee /root/jk03_play.log

tail -n 80 /root/jk03_play.log
```

### 云服务器 Desktop 很卡

这是正常现象。Desktop 是远程视频流，Isaac Sim GUI 又很吃资源。训练时使用：

```text
--headless
```

只在短时间键盘控制时打开 Desktop。想看效果时，优先 headless 录视频再下载到 Mac。

### 找不到 checkpoint

查找所有模型：

```bash
find logs/rsl_rl -name "model_*.pt" | sort | tail -20
```

把找到的路径填到 `--checkpoint` 后面。

### 环境名找不到

重新列出 JK03 环境：

```bash
/root/IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py --keyword JK03
```

本仓库当前使用的环境名是：

```text
RobotLab-Isaac-Velocity-Flat-JK03-v0
RobotLab-Isaac-Velocity-Rough-JK03-v0
```

## 12. 更多文档

```text
robot_lab-main/docs/jk03_pretrain_checklist_cn.md
robot_lab-main/docs/jk03_cloud_training_4090_cn.md
rl_sar-main/docs/manual_mujoco_jk03_cn.md
```

# zhexi-chen

deeplearning for dog robot
made by Andy Chen
The data of dog robot is from yuanjie ruizhi company
