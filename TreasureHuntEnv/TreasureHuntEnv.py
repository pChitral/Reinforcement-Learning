import streamlit as st
import matplotlib.pyplot as plt
import numpy as np 

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
