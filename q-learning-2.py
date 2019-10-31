import gym
import numpy as np

env = gym.make("MountainCar-v0")
# env.reset()

LEARNING_RATE = 0.1
# Discount - How important are future actions/rewards over current actions/rewards
DISCOUNT = 0.95
EPISODES = 25000

# Let us know what's happening every 2000 episodes
SHOW_EVERY = 2000

# Higher the epsilon, more likely we are to perform a random action and explore actions. 
# Over time, we want model to stop exploring. 
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OBSSPACE_SIZE = [20] * len(env.observation_space.high)
discrete_obsspace_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSSPACE_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSSPACE_SIZE + [env.action_space.n]))

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_obsspace_win_size
	return tuple(discrete_state.astype(np.int))

#Running over episodes
for episode in range(EPISODES):
	if(episode % SHOW_EVERY == 0):
		print(episode)
		render = True
	else:
		render = False

	discrete_state = get_discrete_state(env.reset())
	done = False

	# print(discrete_state)
	# print(q_table[discrete_state])
	# Best action to take from Q-Table
	# print(np.argmax(q_table[discrete_state]))

	while not done:
		# While this works fine, once it learns a method, it sticks and tries to optimize along the same path
		# action = np.argmax(q_table[discrete_state])

		# Here we try to add some randomness to it
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)			

		# new_state = position, velocity of the car!
		new_state, reward, done, _ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)

		# Rendering only every SHOW_EVERY times
		if render:
			env.render()
		if not done:
			# Backpropagating max future Q-Value to learn
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]
			# Only usually get a reward after finally getting a future_q, otherwise we just get -1
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action,)] = new_q
		elif new_state[0] >= env.goal_position:
			print(f"We made it on episode {episode}")
			q_table[discrete_state + (action,)] = 0

		discrete_state = new_discrete_state

	if (END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING):
		epsilon -= epsilon_decay_value

env.close()