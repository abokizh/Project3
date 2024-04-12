import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, learning_rate, discount_factor, epsilon, action_size, state_size):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.action_size = action_size
        self.state_size = state_size
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state, :])
    
    def learn(self, state, action, reward, next_state, done):
        target = reward + (0 if done else self.discount_factor * np.max(self.q_table[next_state, :]))
        error = target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * error
    
    def decay_epsilon(self, episode, total_episodes):
        self.epsilon -= self.epsilon / total_episodes

# Initialize environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# Q-Learning parameters
learning_rate = 0.8
discount_factor = 0.95
initial_epsilon = 1.0
total_episodes = 2000
action_size = env.action_space.n
state_size = env.observation_space.n

# Initialize Q-Learning agent
agent = QLearningAgent(learning_rate, discount_factor, initial_epsilon, action_size, state_size)

# Training loop
rewards = []
for episode in tqdm(range(total_episodes), desc="Training"):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    agent.decay_epsilon(episode, total_episodes)
    rewards.append(total_reward)

# Visualize the training rewards
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Evaluate the trained agent
env.close()
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='human')
for episode in range(3):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = np.argmax(agent.q_table[state, :])
        next_state, reward, done, truncated, info = env.step(action)
        env.render()
        state = next_state
        total_reward += reward
    
    print(f"Episode {episode} finished with reward: {total_reward}")
    input("Press Enter to continue to the next episode...")

env.close()
