# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import toml
import inspect

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##


def _patch_rsl_rl_actor_critic_cfg() -> None:
    """Keep robot_lab configs importable across nearby Isaac Lab / RSL-RL versions."""
    try:
        from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
    except Exception:
        return

    init_signature = inspect.signature(RslRlPpoActorCriticCfg.__init__)
    unsupported_normalization_args = {
        "actor_obs_normalization",
        "critic_obs_normalization",
    } - set(init_signature.parameters)
    if not unsupported_normalization_args:
        return

    original_init = RslRlPpoActorCriticCfg.__init__

    def compatible_init(self, *args, **kwargs):
        for name in unsupported_normalization_args:
            kwargs.pop(name, None)
        original_init(self, *args, **kwargs)

    RslRlPpoActorCriticCfg.__init__ = compatible_init


_patch_rsl_rl_actor_critic_cfg()


# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
