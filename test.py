import gym
import custom_gym_envs

# env = gym.make("FetchReachEnv-v999")
env = gym.make("FetchReachEnv-v1")

# action space: Box(-1.0, 1.0, (4,), float32)
# observation_space["observation"]: Box(-inf, inf, (25,), float32)
# observation_space["achieved_goal"]: Box(-inf, inf, (3,), float32)
# env.observation_space["desired_goal"]: Box(-inf, inf, (3,), float32)

for e in range(5):
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()
