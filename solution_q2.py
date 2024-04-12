import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
#2.1 Function to Initialize the gym environment
def create_env(render_mode=None):
    return gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=render_mode)
# Using 2 different render modes for faster simulation and policy evaluation

#2.5 Function to run the policy in the environment and render the GUI
def run_policy(env, policy, num_episodes=3, render_mode=None):
    total_rewards = []
    for episode in range(num_episodes):
        env.close()  # Close the previous environment instance if open
        env = create_env(render_mode)  # Create a new environment with the specified render mode
        observation = env.reset()
        state = observation if isinstance(observation, int) else observation[0]  # Ensure state is an integer
        total_reward = 0
        done = False
        while not done:
            action = int(policy[state])  # Make sure to convert action to int if necessary
            observation, reward, done, truncated, info = env.step(action)
            state = observation if isinstance(observation, int) else observation[0]  # Ensure state is an integer
            total_reward += reward
            if render_mode == 'human':
                env.render()  # Render the environment's state to the GUI
            if done or truncated:
                break
        total_rewards.append(total_reward)
        #if render_mode == 'human':
            #input("Press Enter to continue to the next episode...")  # Wait for user input just for debugging
    env.close()
    return total_rewards





env = create_env(render_mode='rgb_array')

# Initialize matrices for counting transitions and rewards
transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
reward_sums = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
state_action_count = np.zeros((env.observation_space.n, env.action_space.n))

#2.2
# Simulate the environment with random actions
for episode in range(1000):
    state = env.reset()[0]
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        next_state, reward, done, truncated, info = env.step(action)
        
        # Make sure the indices for the current state and action are integers
        state_idx = int(state)  # Assuming state is a single integer
        action_idx = int(action)  # Assuming action is a single integer
        next_state_idx = int(next_state)  # Assuming next_state is a single integer
        
        # Update your counts and sums here using the integer indices
        transition_counts[state_idx, action_idx, next_state_idx] += 1
        reward_sums[state_idx, action_idx, next_state_idx] += reward
        
        # Update the state for the next iteration
        state = next_state

non_zero_transitions = transition_counts > 0
print("Non-zero transition counts:", transition_counts[non_zero_transitions])
print("Non-zero expected rewards:", reward_sums[non_zero_transitions])


# Calculate the transition probabilities and rewards
transition_probabilities = np.zeros(transition_counts.shape)
expected_rewards = np.zeros(reward_sums.shape)
'''print("Transition Probabilities:")
print(transition_probabilities)

print("Expected Rewards:")
print(expected_rewards)'''

epsilon = 1e-8
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        total_transitions = np.sum(transition_counts[s, a, :])
        transition_probabilities[s, a, :] = transition_counts[s, a, :] / (total_transitions + epsilon)
        total_rewards = state_action_count[s, a]
        expected_rewards[s, a, :] = reward_sums[s, a, :] / (total_rewards + epsilon)

#2.3
# Initialize variables for value iteration
V = np.zeros(env.observation_space.n)
policy = np.zeros(env.observation_space.n, dtype=int)
gamma = 0.99  # discount factor
threshold = 1e-3  # threshold for convergence
delta = threshold
iteration = 0
max_iterations=800
# Value iteration
for i in range(max_iterations):
    delta = 0
    for s in range(env.observation_space.n):
        v = V[s]
        V[s] = max([sum([transition_probabilities[s, a, s_prime] *
                        (expected_rewards[s, a, s_prime] + gamma * V[s_prime])
                         for s_prime in range(env.observation_space.n)])
                    for a in range(env.action_space.n)])
        delta = max(delta, abs(v - V[s]))
    print(f"Iteration {i}: Delta={delta}")
    if delta < threshold:
        break

print(f"Value iteration completed in {i+1} iterations.")

#2.4
# Extract policy
for s in range(env.observation_space.n):
    policy[s] = np.argmax([sum([transition_probabilities[s, a, s_prime] * (expected_rewards[s, a, s_prime] + gamma * V[s_prime]) for s_prime in range(env.observation_space.n)]) for a in range(env.action_space.n)])
print("Policy extraction completed.")

# Run the policy in the environment
env = create_env(render_mode='human')
total_rewards = run_policy(env, policy, num_episodes=100, render_mode='rgb_array')#change mode if needed
# After running the policy to get total_rewards
plt.figure(figsize=(12, 6))
plt.plot(range(len(total_rewards)), total_rewards, marker='o')
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

