from gym.envs.registration import register

register(
    id="AntEnv-v0",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v1",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v1_BrokenSeveredLeg:AntEnvV1",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v2",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v2_Hip4ROM:AntEnvV2",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v3",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v3_Ankle4ROM:AntEnvV3",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v4",
    entry_point="custom_gym_envs.envs.ant.AntEnv_v4_BrokenUnseveredLeg:AntEnvV4",
    max_episode_steps=1000,
)


# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v0_Normal.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v1",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v1_BrokenShoulderLiftJoint.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v2",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v2_BrokenElbowFlexJoint.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v3",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v3_BrokenWristFlexJoint.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v4",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v4_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v5",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v4_BrokenJointsTBD.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

# FetchReachReachable

register(
    id="FetchReachReachableEnv-v0",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v0_NormalReachable.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachReachableEnv-v1",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v1_BrokenShoulderLiftJointReachable.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachReachableEnv-v2",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v2_BrokenElbowFlexJointReachable.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachReachableEnv-v3",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v3_BrokenWristFlexJointReachable.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v4",
    entry_point="custom_gym_envs.envs.fetchreach.FetchReachEnv_v4_BrokenShoulderLiftSensor.fetch.reach:FetchReachEnv",
    kwargs={"reward_type": "dense"},
    max_episode_steps=50,
)
