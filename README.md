# dog_robot

本仓库包含机器狗训练与仿真代码。现在已经加入一个“不训练、不加载策略模型”的 JK03 本地 MuJoCo 手动仿真入口，直接复用 `jk03` 目录里的 URDF 和 robot_lab 参数，适合先在本机看 3D 模型和轮足关节响应。

```bash
python3 -m pip install mujoco
python3 rl_sar-main/scripts/manual_mujoco_jk03.py
```

默认会读取仓库里的 `rl_sar-main/src/rl_sar/worlds/stairs.world`，把 JK03 放到这个楼梯场景前方。启动后默认站立不动，按 `1` 或 `W/S` 后才开始轮驱。也可以切换到平地或低台阶：

```bash
python3 rl_sar-main/scripts/manual_mujoco_jk03.py
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene ground
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene low-stairs
```

如果移动时想固定看整个楼梯场景：

```bash
python3 rl_sar-main/scripts/manual_mujoco_jk03.py --scene stairs-world --camera fixed
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
