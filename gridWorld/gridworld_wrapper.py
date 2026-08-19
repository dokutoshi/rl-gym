import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# Original observation is a dictionary
env = gym.make('gymnasium_env/GridWorld-v0')
env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))

obs, info = env.reset()
obs
{'agent': array([4, 1]), 'target': array([2, 4])}

# Wrap it to flatten observations into a single array
wrapped_env = FlattenObservation(env)
wrapped_env.observation_space
Box(0, 4, (4,), int64)

obs, info = wrapped_env.reset()
obs
array([3, 0, 2, 1])  # [agent_x, agent_y, target_x, target_y]