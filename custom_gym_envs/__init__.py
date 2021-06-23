from gym.envs.registration import register

register(
    id='AntEnv-v0',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v0_Normal:AntEnvV0',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v1',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v1_BrokenSeveredLeg:AntEnvV1',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v2',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v2_Hip4ROM:AntEnvV2',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v3',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v3_Ankle4ROM:AntEnvV3',
    max_episode_steps=1000,
)

register(
    id='AntEnv-v4',
    entry_point='custom_gym_envs.envs.ant.AntEnv_v4_BrokenUnseveredLeg:AntEnvV4',
    max_episode_steps=1000,
)


# FetchPickAndPlace

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }
    register(
        id='FetchReachEnv{}-v0'.format(suffix),
        entry_point='custom_gym_envs.envs.FetchReach.FetchReachEnv_v0_Normal.fetch.reach:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchReachEnv{}-v1'.format(suffix),
        entry_point='custom_gym_envs.envs.FetchReach.FetchReachEnv_v1_BrokenShoulderLiftJoint.fetch.reach:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchReachEnv{}-v2'.format(suffix),
        entry_point='custom_gym_envs.envs.FetchReach.FetchReachEnv_v2_BrokenElbowFlexJoint.fetch.reach:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchReachEnv{}-v3'.format(suffix),
        entry_point='custom_gym_envs.envs.FetchReach.FetchReachEnv_v3_BrokenWristFlexJoint.fetch.reach:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
    register(
        id='FetchReachEnv{}-v4'.format(suffix),
        entry_point='custom_gym_envs.envs.FetchReach.FetchReachEnv_v4_BrokenGrip.fetch.reach:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )