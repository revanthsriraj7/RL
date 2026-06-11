import gymnasium as gym

env = gym.make("CartPole-v1")
print(f"action space: {env.action_space}")

print(f"action space sample: {env.action_space.sample()}")

print(f"obs space: {env.observation_space}")

print(f"obs space sample: {env.observation_space.sample()}")
