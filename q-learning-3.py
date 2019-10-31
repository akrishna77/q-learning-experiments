import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
# env.reset()

LEARNING_RATE = 0.1
# Discount - How important are future actions/rewards over current actions/rewards
DISCOUNT = 0.95
EPISODES = 2000

# Let us know what's happening every 2000 episodes
SHOW_EVERY = 500
STATS_EVERY = 100

# Higher the epsilon, more likely we are to perform a random action and explore actions. 
# Over time, we want model to stop exploring. 
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


DISCRETE_OBSSPACE_SIZE = [20] * len(env.observation_space.high)
discrete_obsspace_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSSPACE_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSSPACE_SIZE + [env.action_space.n]))

# List of rewards
ep_rewards = []
# Aggregate rewards dictionary
aggr_ep_rewards = {'ep' : [], 'avg' : [], 'min' : [], 'max' : []}

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_obsspace_win_size
	return tuple(discrete_state.astype(np.int))

#Running over episodes
for episode in range(EPISODES):
	episode_reward = 0
	if(episode % SHOW_EVERY == 0):
		print(episode)
		render = True
	else:
		render = False

	discrete_state = get_discrete_state(env.reset())
	done = False

	while not done:
		# While this works fine, once it learns a method, it sticks and tries to optimize along the same path
		# action = np.argmax(q_table[discrete_state])

		# Here we try to add some randomness to it
		if np.random.random() > epsilon:
            # Get best action from Q table
			action = np.argmax(q_table[discrete_state])
		else:
            # Get random action
			action = np.random.randint(0, env.action_space.n)			

		# new_state = position, velocity of the car!
		new_state, reward, done, _ = env.step(action)
		episode_reward += reward
		new_discrete_state = get_discrete_state(new_state)

		# Rendering only every SHOW_EVERY times
		if render:
			env.render()

        # If simulation did not end yet after last step - update Q table
		if not done:
			# Backpropagating max future Q-Value to learn
            # Maximum possible Q value in next step (for new state)
			max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
			current_q = q_table[discrete_state + (action,)]

			# Only usually get a reward after finally getting a future_q, otherwise we just get -1
            # And here's our equation for a new Q value for current state and action
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
			q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reason) - if goal position is achived - update Q value with reward directly
		elif new_state[0] >= env.goal_position:
			print(f"We made it on episode {episode}")
			q_table[discrete_state + (action,)] = 0

		discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
	if (END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING):
		epsilon -= epsilon_decay_value

	ep_rewards.append(episode_reward)

	if not episode % SHOW_EVERY:
		average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
		aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
		print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

	# if episode % 10 == 0:
 #        np.save(f"qtables/{episode}-qtable.npy", q_table)

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()