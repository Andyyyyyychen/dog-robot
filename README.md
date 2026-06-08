# dog_robot

本仓库包含机器狗训练与仿真代码。现在已经加入一个“不训练、不加载策略模型”的 JK03 本地 MuJoCo 手动仿真入口，直接复用 `jk03` 目录里的 URDF 和 robot_lab 参数，适合先在本机看 3D 模型和轮足关节响应。

```bash
python3 -m pip install mujoco
python3 rl_sar-main/scripts/manual_mujoco_jk03.py
```

常用按键：

```text
1 开始轮驱    0 站立       W/S 前进/后退
A/D 左右偏置  Q/E 转向      Space 清空指令
R 重置姿态    H 帮助        Esc 退出
```

详细说明见 `rl_sar-main/docs/manual_mujoco_jk03_cn.md`。
# zhexi-chen
deeplearning for dog robot
made by Andy Chen
The data of dog robot is from yuanjie ruizhi company
