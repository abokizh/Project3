import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import gymnasium as gym
env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
###YOUR Q-LEARNING CODE BEGINS

## Properties
episodes = 1000000
rate = 0.001
discount = 0.87
exploration_value = 1.0
exploration_terminal = 0.1
exploration_discount = exploration_value/(episodes/3) 

## RL Agent
class Agent:

	# Init
    def __init__(self, rate: float, discount: float, exploration_value: float, exploration_terminal: float, exploration_discount: float):
        self.table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.rate = rate
        self.discount = discount
        self.exploration_value = exploration_value
        self.exploration_terminal = exploration_terminal
        self.exploration_discount = exploration_discount

    # Make the optimal move, or explore
    def move(self, observation: tuple[int, int, bool]) -> int:
        if self.exploration_value > np.random.random():
            return env.action_space.sample()
        else:
            return int(np.argmax(self.table[observation]))

    # Update the q-value table
    def update(self, observation: tuple[int, int, bool], action: int, reward: float, terminated: bool, next_obs: tuple[int, int, bool]):
        new_table = (not terminated) * np.max(self.table[next_obs])
        error = reward + self.discount * new_table - self.table[observation][action]
        self.table[observation][action] = self.rate * error + self.table[observation][action]

    # Decrease exploration each time, so agent acts optimally more by time
    def decrease_exploration(self):
        self.exploration_value = max(self.exploration_terminal, self.exploration_value - exploration_discount)

agent = Agent(rate=rate, discount=discount, exploration_value=exploration_value, exploration_terminal=exploration_terminal, exploration_discount=exploration_discount)

## Run through episodes for RL
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=episodes)
# total_reward = 0
for episode in range(episodes):

    game_over = False
    observation, info = env.reset()

    while not game_over:
        action = agent.move(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        agent.update(observation, action, reward, terminated, new_observation)
        # total_reward += reward
        observation = new_observation
        game_over = terminated or truncated

    agent.decrease_exploration()
# print("Total reward: ", total_reward)
# Plot the results
rolling_length = 500
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
plt.tight_layout()
plt.show()
###YOUR Q-LEARNING CODE ENDS