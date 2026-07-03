# JK04 Four-Wheel Yaw Design

## Goal

Make `RobotLab-Isaac-Velocity-Flat-Yaw-JK04-v0` learn flat in-place yaw with all four wheel joints participating, instead of satisfying yaw tracking mostly with the rear wheels.

## Evidence

`model_2100.pt` reaches yaw velocity but has poor front/rear participation:

- `yaw=+0.45`: front/rear wheel action ratio `0.43`, wheel velocity ratio `0.46`.
- `yaw=-0.45`: front/rear wheel action ratio `0.11`, wheel velocity ratio `0.21`.

This shows the action mapping is live, but the reward permits a rear-wheel-dominant local optimum.

## Approach

Use a JK04-local reward term, not a shared MDP reward edit, so unrelated JK03 and shared reward changes stay untouched. The new reward is active only for near-in-place yaw commands and multiplies three signals:

- signed yaw progress in the commanded direction,
- front and rear wheel speed magnitude,
- front/rear balance ratio.

The yaw stage will also enable the existing wheel yaw rewards and reduce pure yaw progress weight. Stepping rewards are disabled for this first wheel-differential stage because the target behavior is four wheels rolling on flat ground.

## Success Criteria

- The JK04 yaw environment still creates and trains for one PPO iteration.
- Active rewards include the new front/rear wheel participation reward.
- A retrained checkpoint should move front/rear action and velocity ratios much closer to `1.0` than `model_2100.pt`.
