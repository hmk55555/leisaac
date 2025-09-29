import gymnasium as gym

gym.register(
    id='LeIsaac-SO101-CustomBiArm-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_bi_arm_env_cfg:CustomBiArmEnvCfg",
    },
)



