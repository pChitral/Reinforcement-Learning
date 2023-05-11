import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import optuna


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
# Create an instance of the environment and agent


class TreasureHuntEnv():
    def __init__(self):
        self.action_space = np.array([0, 1, 2, 3])
        self.observation_space = np.array(
            [(i, j) for i in range(4) for j in range(4)])
        self.state = st.sidebar.selectbox("Select starting position", [
                                          (i, j) for i in range(4) for j in range(4)])
        self.treasure = st.sidebar.selectbox("Select treasure position", [
                                             (i, j) for i in range(4) for j in range(4)])
        self.obstacle = st.sidebar.multiselect("Select penalty positions", [
                                               (i, j) for i in range(4) for j in range(4)])
        self.reward = self.get_reward()

    def get_reward(self):
        reward = {}
        for i in range(4):
            for j in range(4):
                if (i, j) in self.obstacle:
                    reward[(i, j)] = -5
                elif (i, j) == self.treasure:
                    reward[(i, j)] = 10
                else:
                    reward[(i, j)] = -1
        return reward

    def step(self, action):
        reward = self.reward.get(self.state, 0)
        if self.state == self.treasure:
            done = True
        else:
            done = False
        if action == 0:
            next_state = (self.state[0] - 1, self.state[1])
        elif action == 1:
            next_state = (self.state[0] + 1, self.state[1])
        elif action == 2:
            next_state = (self.state[0], self.state[1] + 1)
        elif action == 3:
            next_state = (self.state[0], self.state[1] - 1)
        else:
            raise ValueError("Invalid action")
        if (next_state[0] >= 0 and next_state[0] < 4 and
                next_state[1] >= 0 and next_state[1] < 4):
            if next_state not in self.obstacle:
                self.state = next_state
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = (0, 0)
        return np.array(self.state)

    def render(self):
        grid = np.full((4, 4), -2)

        # mark the current position of the agent
        grid[self.state] = 1
        # mark the treasure location
        grid[self.treasure] = 2
        # mark the obstacle locations
        for obs in self.obstacle:
            grid[obs] = -1
        # plot the grid
        plt.imshow(grid)
        plt.show()


class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return random.choice(self.action_space)

    def reset(self):
        pass


env = TreasureHuntEnv()
agent = QLearningAgent(env.action_space, env.observation_space)

# Define the Streamlit app


def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Treasure Hunt Q-Learning Agent")

    # Reset the environment
    obs = env.reset()
    done = False
    reward = 0
    count = 1

    # Run the simulation
    while not done:
        # Get the agent's action
        action = agent.act(obs, reward, done)

        # Take the action and get the next observation and reward
        next_obs, reward, done, _ = env.step(action)
        if done:
            break

        # Get the next action
        next_action = agent.act(next_obs, reward, done)

        # Update the Q-table
        agent.learn(obs, action, reward, next_obs, next_action, done)

        # Update the current observation
        obs = next_obs

        # Display the action, reward, and rendering
        st.write(f"For the {count}th step")
        st.write('Action:', action)
        st.write('Reward:', reward)
        st.write('Done:', done)
        env.render()
        st.pyplot()

        count += 1
    if count>1: 
        st.balloons()
        st.success('We have reached our treasure!', icon="✅")

# Run the app
if __name__ == '__main__':
    app()
