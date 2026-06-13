# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


@dataclass
class JK03RoughTrainerCfg(TrainerCfg):
    max_iterations = 20000
    save_interval = 100
    experiment_name = "jk03_rough"
    agent_factory = cusrl.ActorCritic.Factory(
        # Match the RSL-RL JK03 PPO setup: longer rollouts and smaller updates
        # are less likely to destroy stair locomotion while the policy explores.
        num_steps_per_update=32,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 3.0e-4}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=4, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.006),
            cusrl.hook.GradientClipping(max_grad_norm=0.5),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.006),
        ],
    )


@dataclass
class JK03FlatTrainerCfg(JK03RoughTrainerCfg):
    max_iterations = 5000
    experiment_name = "jk03_flat"
