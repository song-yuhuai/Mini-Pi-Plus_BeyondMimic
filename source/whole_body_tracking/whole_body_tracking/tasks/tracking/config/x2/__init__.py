import gymnasium as gym

from . import agents, chassis_env_cfg, flat_env_cfg

##
# Register Gym environments for X2 flat tracking.
##

gym.register(
    id="Tracking-Flat-X2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.X2BaseEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatPPORunnerCfg",
    },
)



gym.register(
    id="Tracking-Chassis-X2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": chassis_env_cfg.X2ChassisEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-X2-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.X2RobustEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatRobustPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Chassis-X2-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": chassis_env_cfg.X2ChassisRobustEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatRobustPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-X2-Simple-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.X2SimpleRobustEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatRobustPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Chassis-X2-Simple-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": chassis_env_cfg.X2ChassisSimpleRobustEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatRobustPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-X2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.X2BasePlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Chassis-X2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": chassis_env_cfg.X2ChassisPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-X2-Robust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.X2RobustPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatRobustPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Chassis-X2-Robust-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": chassis_env_cfg.X2ChassisRobustPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2FlatRobustPPORunnerCfg",
    },
)
