import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random


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
        st.pyplot()


class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return random.choice(self.action_space)

    def reset(self):
        pass


def main():
    st.title("Treasure Hunt Game")
    st.subheader("Using a Random Agent")

    env = TreasureHuntEnv()
    agent = RandomAgent(env.action_space)

    obs = env.reset()
    done = False
    reward = 0  # initialize reward to 0
    count = 1
    while not done:
        action = agent.act(obs, reward, done)
        obs, reward, done, _ = env.step(action)

        # render the game
        st.write(f"**Step {count}**")
        grid = np.full((4, 4), -2)
        grid[obs] = 1  # mark the current position of the agent
        grid[env.treasure] = 2  # mark the treasure location
        for obs_loc in env.obstacle:
            grid[obs_loc] = -1  # mark the obstacle locations
        plt.imshow(grid, cmap='viridis')
        st.pyplot(plt)

        count += 1

    st.write("🎉 Congrats! You have found the treasure!")


if __name__ == '__main__':
    main()
