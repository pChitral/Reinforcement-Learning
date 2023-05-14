# Treasure Hunt Environment and Random Agent

This code defines a grid-world environment called TreasureHuntEnv and an agent called RandomAgent that performs random actions. The environment is a 4x4 grid where the agent starts at (0, 0) and has to find the treasure located at (2, 2). There are obstacles at positions (0, 2), (1, 2), (2, 0), and (3, 1), and the agent receives a reward of -1 for each step taken. If the agent reaches the treasure, it receives a reward of 10, and if it collides with an obstacle, it receives a reward of -5.

The TreasureHuntEnv class has the following methods:

- `__init__(self)`: Initializes the environment by defining the action space, observation space, starting state, treasure location, obstacle locations, and reward dictionary.
- `step(self, action)`: Executes an action in the environment and returns the next state, reward, done, and info. The next state is the result of taking the specified action in the current state. The reward is the reward received for the action. The done flag is set to True when the agent reaches the treasure, and False otherwise. The info dictionary is currently empty.
- `reset(self)`: Resets the environment to its initial state and returns the initial state.
- `render(self)`: Renders the current state of the environment as a plot using the matplotlib library.

The RandomAgent class has the following methods:

- `__init__(self, action_space)`: Initializes the agent by defining the action space.
- `act(self, observation, reward, done)`: Chooses a random action from the action space.
- `reset(self)`: This method resets the agent to its initial state.

The main part of the code creates an instance of the environment and the agent and runs a loop that performs actions in the environment until the agent reaches the treasure or 20 steps have been taken. In each step, the agent selects a random action, and the environment returns the next state, reward, and done flag. The state, reward, and done flag are printed, and the environment is rendered as a plot.

# SARSA Agent

This code defines a SARSA agent called SARSAAgent that learns to navigate the grid-world environment. The agent uses the SARSA algorithm to update its Q-table and chooses actions based on the Q-values. The agent starts with a Q-table of zeros and updates the Q-values using the following formula:

Q(s,a) = Q(s,a) + alpha _ (R + gamma _ Q(s',a') - Q(s,a))

where Q(s,a) is the Q-value for state s and action a, alpha is the learning rate, R is the reward received for the action, gamma is the discount factor, s' is the next state, a' is the next action, and done is a flag indicating whether the episode has ended.

The SARSAAgent class has the following methods:

- **init**(self, action_space, observation_space, alpha=0.1, gamma=0.9, epsilon=0.1): Initializes the agent by defining the action space, observation space, learning rate, discount factor, and exploration rate. It also creates a Q-table of zeros with dimensions (len(observation_space), len(action_space)).
- act(self, observation, reward, done): Chooses an action based on the epsilon-greedy policy. If a random number is less than epsilon, the agent chooses a random action. Otherwise, it chooses the action with the highest Q-value.
- learn(self, state, action, reward, next_state, next_action, done): Updates the Q-values in the Q-table using the SARSA algorithm. The Q-values represent the expected discounted reward for a particular state-action pair. The SARSA algorithm is an on-policy control algorithm that uses the current policy to select actions and updates the Q-values accordingly. The learn() method in the SARSAAgent class implements the SARSA algorithm and updates the Q-values using the following formula:

`Q(s,a) = Q(s,a) + alpha * (R + gamma * Q(s',a') - Q(s,a))`

- where Q(s,a) is the Q-value for state s and action a, alpha is the learning rate, R is the reward received for the action, gamma is the discount factor, s' is the next state, a' is the next action, and done is a flag indicating whether the episode has ended.

- In the learn() method, the Q-value for the current state-action pair is updated using the SARSA algorithm, which takes into account the immediate reward and the discounted Q-value for the next state-action pair. The new Q-value is then updated using the learning rate, which controls the amount by which the Q-value is updated.

- After the Q-value update, the exploration-exploitation tradeoff parameter (epsilon) is updated using the epsilon decay factor, which reduces the value of epsilon over time to encourage exploitation over exploration. This update is only done if the current episode is completed (i.e., the done flag is set to True).

- The SARSA algorithm is an effective and widely used algorithm in reinforcement learning for solving control problems. However, it has some limitations, such as being more computationally expensive than other algorithms and requiring a relatively large amount of data to learn the optimal policy. Nonetheless, it has been successfully applied to various real-world problems, such as robotics, game playing, and autonomous driving.
