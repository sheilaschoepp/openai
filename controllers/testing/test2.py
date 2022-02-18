import gym
import time

episodes = 1
steps = 1000

# normal

start = time.time()

env = gym.make("Ant-v2")

for e in range(episodes):
    env.seed(3)
    env.action_space.seed(e)
    env.reset()
    for s in range(steps):
        env.step(env.action_space.sample())
        env.render()
env.close()

end = time.time()

normal_time = end - start

# slowdown

start = time.time()

env = gym.make("Ant-v2")

for e in range(episodes):
    env.seed(3)
    env.action_space.seed(e)
    env.reset()
    for s in range(steps):
        env.step(env.action_space.sample())
        env.render()
        time.sleep(0.1)
env.close()

end = time.time()

slowed_time = end - start

slowdown = slowed_time / normal_time

print(slowdown)

# 8.755432816836493