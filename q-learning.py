import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
#Discount - How important are future actions/rewards over current actions/rewards
DISCOUNT = 0.95
EPISODES = 25000

# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

DISCRETE_OBSSPACE_SIZE = [20] * len(env.observation_space.high)
discrete_obsspace_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSSPACE_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSSPACE_SIZE + [env.action_space.n]))
# print(q_table)
# print(q_table.shape)

done = False

while not done:
	action = 2
	new_state, reward, done, _ = env.step(action)
	# print(reward, new_state)
	env.render()

env.close()