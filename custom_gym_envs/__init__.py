from gym.envs.registration import register

register(
    id='AntEnv-v0',
    entry_point='custom_gym_envs.envs.AntEnv_v0_normal:AntEnvV0',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v1',
    entry_point='custom_gym_envs.envs.AntEnv_v1_brokenleg:AntEnvV1',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v2',
    entry_point='custom_gym_envs.envs.AntEnv_v2_hip4rom:AntEnvV2',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v3',
    entry_point='custom_gym_envs.envs.AntEnv_v3_ankle4rom:AntEnvV3',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v4',
    entry_point='custom_gym_envs.envs.AntEnv_v4_ab_addedlink:AntEnvV4',
    max_episode_steps=1000,
)