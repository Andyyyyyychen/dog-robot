# dog_robot

本仓库包含机器狗训练与仿真代码。现在已经加入一个“不训练、不加载策略模型”的 Go2 本地 MuJoCo 手动仿真入口，适合先在 Linux 本机看 3D 模型和关节运动。

```bash
python3 -m pip install mujoco
python3 rl_sar-main/scripts/manual_mujoco_go2.py
```

常用按键：

```text
1 开始小跑    0 站立       W/S 前进/后退
A/D 左右偏置  Q/E 转向      Space 清空速度
R 重置姿态    H 帮助        Esc 退出
```

详细说明见 `rl_sar-main/docs/manual_mujoco_go2_cn.md`。
# zhexi-chen
deeplearning for dog robot
made by Andy Chen
The data of dog robot is from yuanjie ruizhi company