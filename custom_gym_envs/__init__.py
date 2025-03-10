from gymnasium.envs.registration import register

register(
    id="Ant-F0",
    entry_point="custom_gym_envs.envs.ant.AntEnvF0_Normal:AntEnvF0",
    max_episode_steps=1000,
)

register(
    id="Ant-F1",
    entry_point="custom_gym_envs.envs.ant.AntEnvF1_Ankle4ROM:AntEnvF1",
    max_episode_steps=1000,
)

register(
    id="Ant-F2",
    entry_point="custom_gym_envs.envs.ant.AntEnvF2_Hip4ROM:AntEnvF2",
    max_episode_steps=1000,
)

register(
    id="Ant-F3",
    entry_point="custom_gym_envs.envs.ant.AntEnvF3_BrokenSeveredLimb:AntEnvF3",
    max_episode_steps=1000,
)

register(
    id="Ant-F4",
    entry_point="custom_gym_envs.envs.ant.AntEnvF4_BrokenUnseveredLimb:AntEnvF4",
    max_episode_steps=1000,
)


# FetchReach

register(
    id="FetchReach-F0",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachF0_Normal.reach:MujocoFetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReach-F1",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachF1_FrozenShoulderLiftPositionSensor.reach:MujocoFetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReach-F2",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachF2_SlipperyElbowFlexJoint.reach:MujocoFetchReachEnv",
    max_episode_steps=50,
)
