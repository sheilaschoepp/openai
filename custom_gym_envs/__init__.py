from gymnasium.envs.registration import register

register(
    id="AntEnv-F0-N",
    entry_point="custom_gym_envs.envs.ant.AntEnvF0_Normal:AntEnvF0",
    max_episode_steps=1000,
)

register(
    id="AntEnvF1-AROM",
    entry_point="custom_gym_envs.envs.ant.AntEnvF1_Ankle4ROM:AntEnvF1",
    max_episode_steps=1000,
)

register(
    id="AntEnvF2-HROM",
    entry_point="custom_gym_envs.envs.ant.AntEnvF2_Hip4ROM:AntEnvF2",
    max_episode_steps=1000,
)

register(
    id="AntEnvF3-BSL",
    entry_point="custom_gym_envs.envs.ant.AntEnvF3_BrokenSeveredLimb:AntEnvF3",
    max_episode_steps=1000,
)

register(
    id="AntEnvF4-BUL",
    entry_point="custom_gym_envs.envs.ant.AntEnvF4_BrokenUnseveredLimb:AntEnvF4",
    max_episode_steps=1000,
)


# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v0_Normal.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v1",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v1_BrokenShoulderLiftJoint.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v2",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v2_BrokenElbowFlexJoint.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v3",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v3_BrokenWristFlexJoint.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v4",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v4_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v5",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v5_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
register(
    id="FetchReachEnv-v6",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v6_ElbowFlexNoisyMovement.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
