from gym.envs.registration import register

register(
    id='AntEnv-v0',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v0_normal:AntEnvV0',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v1',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v1_brokenleg:AntEnvV1',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v2',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v2_hip4rom:AntEnvV2',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v3',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v3_ankle4rom:AntEnvV3',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v4',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v4_ab_addedlink:AntEnvV4',
    max_episode_steps=1000,
)

register(
    id='FetchPickAndPlaceEnv-v0',
    entry_point='custom_gym_envs.envs.fetchpickandplace.FetchPickAndPlaceEnv_v0_normal:FetchPickAndPlaceEnvV0',
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlaceEnv-v1',
    entry_point='custom_gym_envs.envs.fetchpickandplace.FetchPickAndPlaceEnv_v1:FetchPickAndPlaceEnvV1',
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlaceEnv-v2',
    entry_point='custom_gym_envs.envs.fetchpickandplace.FetchPickAndPlaceEnv_v2:FetchPickAndPlaceEnvV2',
    max_episode_steps=50,
)

register(
    id='FetchPickAndPlaceEnv-v3',
    entry_point='custom_gym_envs.envs.fetchpickandplace.FetchPickAndPlaceEnv_v3:FetchPickAndPlaceEnvV3',
    max_episode_steps=50,
)

register(
    id='FetchReachEnv-v0',
    entry_point='custom_gym_envs.envs.fetchreach.FetchReachEnv_v0_normal:FetchReachEnvV0',
    max_episode_steps=50,
)

register(
    id='FetchReachEnv-v1',
    entry_point='custom_gym_envs.envs.fetchreach.FetchReachEnv_v1:FetchReachEnvV1',
    max_episode_steps=50,
)