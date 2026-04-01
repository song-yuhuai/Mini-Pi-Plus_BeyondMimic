from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import DelayBuffer, configclass


class DelayedJointPositionAction(JointPositionAction):
    """Joint position action with a control-step delay buffer."""

    cfg: "DelayedJointPositionActionCfg"

    def __init__(self, cfg: "DelayedJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._processed_actions_delay_buffer = DelayBuffer(cfg.max_delay, self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = self._processed_actions_delay_buffer.compute(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        else:
            num_envs = len(env_ids)

        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self.device,
        )
        self._processed_actions_delay_buffer.set_time_lag(time_lags, env_ids)
        self._processed_actions_delay_buffer.reset(env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for joint position control with control-step delay."""

    class_type: type = DelayedJointPositionAction

    min_delay: int = 0
    """Minimum control delay in environment control steps."""

    max_delay: int = 0
    """Maximum control delay in environment control steps."""
