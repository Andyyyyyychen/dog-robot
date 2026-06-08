# JK03 RTX 4090 云服务器训练步骤

这份文档用于把 JK03 的深度学习训练放到云服务器 RTX 4090 / 24GB 上完成。本地电脑只负责看 MuJoCo、检查数据、修改代码和推送 GitHub；正式训练在云端进行。

## 1. 云服务器规格

租服务器时优先选择：

```text
GPU: RTX 4090 24GB
系统: Ubuntu 22.04 或 Ubuntu 24.04
磁盘: 至少 150GB，推荐 200GB+
内存: 至少 32GB，推荐 64GB
镜像: 优先选择 NVIDIA Driver / CUDA / PyTorch 已安装的镜像
```

RTX 4090 的 24GB 显存很强，但不是无限显存。训练时 `--num_envs` 建议先从 `256` 或 `512` 试，稳定后再加到 `1024`。

## 2. 登录云服务器

在你的本地电脑终端里执行：

```bash
ssh 用户名@服务器IP
```

登录后先检查显卡：

```bash
nvidia-smi
```

如果能看到 RTX 4090、Driver Version、CUDA Version，说明 GPU 驱动已经好了。

如果 `nvidia-smi` 不存在或报错，先不要训练。很多云服务器可以直接换一个 CUDA / PyTorch / NVIDIA Driver 已安装的镜像，这是最省事的方式。如果必须自己装驱动，可以尝试：

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu
sudo reboot
```

重启后重新登录，再跑：

```bash
nvidia-smi
```

## 3. 安装基础工具

```bash
sudo apt update
sudo apt install -y git git-lfs wget curl build-essential cmake tmux htop
```

建议训练都放进 `tmux`，这样 SSH 断开后训练不会停：

```bash
tmux new -s jk03
```

临时退出但不停止训练：

```text
Ctrl + B
然后按 D
```

重新进入：

```bash
tmux attach -t jk03
```

## 4. 用 git clone 拉你的仓库

不要用 zip。云服务器上执行：

```bash
cd ~
git clone https://github.com/Andyyyyyychen/dog-robot.git dog_robot
cd ~/dog_robot
git status
```

先检查 JK03 数据：

```bash
python3 robot_lab-main/scripts/tools/check_jk03_pretrain.py --skip-mujoco
```

看到 `0 failure(s)` 就说明 JK03 数据和训练入口的静态检查没问题。

## 5. 安装 Miniconda

如果云服务器还没有 conda：

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
exec bash
```

创建 Isaac Lab 训练环境：

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
```

以后每次登录云服务器训练前，都先运行：

```bash
conda activate env_isaaclab
```

## 6. 安装 Isaac Sim 和 Isaac Lab

`robot_lab-main` 当前 README 对应 Isaac Lab `v2.3.2`，所以建议安装 Isaac Lab `v2.3.2`。

```bash
conda activate env_isaaclab
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

然后安装 Isaac Lab：

```bash
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git --branch v2.3.2
cd ~/IsaacLab
./isaaclab.sh --install
```

第一次启动 Isaac Sim 可能会下载缓存，时间比较久。如果它问是否接受 NVIDIA EULA，输入：

```text
Yes
```

## 7. 安装你的 robot_lab 工程

```bash
conda activate env_isaaclab
cd ~/dog_robot/robot_lab-main
python -m pip install -e source/robot_lab
```

检查 JK03 环境是否注册成功：

```bash
python scripts/tools/list_envs.py --keyword JK03
```

你必须看到：

```text
RobotLab-Isaac-Velocity-Flat-JK03-v0
RobotLab-Isaac-Velocity-Rough-JK03-v0
```

看到这两个，说明云端训练入口已经通了。

## 8. 小规模试运行

先测试 Flat 环境能不能启动：

```bash
python scripts/tools/zero_agent.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 16
```

再测试随机动作：

```bash
python scripts/tools/random_agent.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 16
```

这两个能跑，说明机器人能生成、仿真能步进、环境没有崩。

## 9. 短训练测试

先跑 100 轮，不要一开始就长时间训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 256 \
  --max_iterations 100
```

另开一个 SSH 窗口看显存：

```bash
watch -n 2 nvidia-smi
```

如果显存还有余量，再增加 `--num_envs`。

## 10. 正式训练 Flat

RTX 4090 / 24GB 可以先试：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --headless \
  --num_envs 1024
```

如果爆显存，改成：

```text
--num_envs 512
```

Flat 训练结果会写到：

```text
~/dog_robot/robot_lab-main/logs/rsl_rl/jk03_flat/
```

## 11. 正式训练 Rough

Flat 稳定后再跑 Rough：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --headless \
  --num_envs 512
```

Rough 比 Flat 更吃显存，先用 `512` 更稳。稳定后再试 `1024`。

Rough 训练结果会写到：

```text
~/dog_robot/robot_lab-main/logs/rsl_rl/jk03_rough/
```

## 12. 训练后验证和导出

平地策略：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-JK03-v0 \
  --num_envs 16
```

复杂地形策略：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-JK03-v0 \
  --num_envs 16
```

`play.py` 会导出：

```text
policy.pt
policy.onnx
```

大概位置是：

```text
logs/rsl_rl/jk03_flat/某次训练/exported/
logs/rsl_rl/jk03_rough/某次训练/exported/
```

## 13. 总顺序

```text
1. 租 RTX 4090 / 24GB 云服务器
2. 登录后确认 nvidia-smi 正常
3. git clone 你的 dog-robot 仓库
4. 安装 conda + Isaac Sim + Isaac Lab v2.3.2
5. pip install -e robot_lab
6. list_envs.py --keyword JK03
7. zero_agent / random_agent 小测试
8. Flat 短训 100 轮
9. Flat 正式训练
10. Rough 正式训练
```

不要修改 JK03 原始数据。如果需要改奖励、动作、课程或新动作技能，后续应该新建派生任务配置，不要直接覆盖当前 `jk03` 初始参数。
