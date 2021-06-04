from gym.envs.registration import register
# Ant

register(
    id='AntEnv-v0',
    entry_point='custom_gym_envs.envs.AntEnv.AntEnv_v0_normal:AntEnvV0',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v1',
    entry_point='custom_gym_envs.envs.AntEnv.AntEnv_v1_brokenleg:AntEnvV1',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v2',
    entry_point='custom_gym_envs.envs.AntEnv.AntEnv_v2_hip4rom:AntEnvV2',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v3',
    entry_point='custom_gym_envs.envs.AntEnv.AntEnv_v3_ankle4rom:AntEnvV3',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v4',
    entry_point='custom_gym_envs.envs.AntEnv.AntEnv_v4_ab_addedlink:AntEnvV4',
    max_episode_steps=1000,
)

# Humanoid

register(
    id='HumanoidEnv-v0',
    entry_point='custom_gym_envs.envs.HumanoidEnv.Humanoid_v0_normal:HumanoidEnvV0',
    max_episode_steps=1000,
)

register(
    id='HumanoidEnv-v1',
    entry_point='custom_gym_envs.envs.HumanoidEnv.Humanoid_v1_faulty:HumanoidEnvV1',
    max_episode_steps=1000,
)

# FetchPickAndPlace

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }
    register(
        id='FetchPickAndPlace{}-v0'.format(suffix),
        entry_point='custom_gym_envs.envs.robotics.fetch.pick_and_place:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=500,
    )
