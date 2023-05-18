import random
import numpy as np


class QLearningAgent():
    def __init__(self, action_space, observation_space, alpha=0.1, gamma=0.9, epsilon=0.1, seed=5):
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_space = action_space
        self.observation_space = observation_space
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.Q = np.zeros(
            (len(observation_space), len(action_space)))  # Q-table
        np.random.seed(seed)
        random.seed(seed)

    def act(self, observation, reward, done):
        if random.uniform(0, 1) < self.epsilon:
            # choose a random action
            return random.choice(self.action_space)
        else:
            # choose the action with the highest Q-value
            state_idx = np.where(
                (self.observation_space == observation).all(axis=1))[0][0]
            q_values = self.Q[state_idx, :]
            if np.all(q_values == 0):
                return random.choice(self.action_space)
            else:
                max_q_value = np.max(q_values)
                max_action_indices = np.where(q_values >= max_q_value)[0]
                return random.choice(max_action_indices)

    def learn(self, state, action, reward, next_state, next_action, done):
        state_idx = np.where(
            (self.observation_space == state).all(axis=1))[0][0]
        next_state_idx = np.where(
            (self.observation_space == next_state).all(axis=1))[0][0]
        q_value = self.Q[state_idx, action]
        next_max_q_value = np.max(self.Q[next_state_idx, :])
        td_target = reward + self.gamma * next_max_q_value * (1 - done)
        td_error = td_target - q_value
        self.Q[state_idx, action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if not done:
            self.Q[next_state_idx, next_action] += self.alpha * \
                (td_target - self.gamma * self.Q[next_state_idx, next_action])

    def reset(self):
        pass
