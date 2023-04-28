import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random


class TreasureHuntEnv:
    def __init__(self, grid_size=10, num_obstacles=10, num_treasures=5, treasure_reward=1):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_treasures = num_treasures
        self.treasure_reward = treasure_reward
        self.grid = np.zeros((grid_size, grid_size))
        self.agent_pos = (0, 0)
        self.obstacle_pos = []
        self.treasure_pos = []
        self.generate_obstacles()
        self.generate_treasures()
        self.action_space = ['up', 'down', 'left', 'right']

    def generate_obstacles(self):
        self.obstacle_pos = []
        for i in range(self.num_obstacles):
            x = np.random.randint(self.grid_size)
            y = np.random.randint(self.grid_size)
            self.grid[x, y] = -1
            self.obstacle_pos.append((x, y))

    def generate_treasures(self):
        self.treasure_pos = []
        for i in range(self.num_treasures):
            x = np.random.randint(self.grid_size)
            y = np.random.randint(self.grid_size)
            while self.grid[x, y] != 0 or (x, y) in self.obstacle_pos:
                x = np.random.randint(self.grid_size)
                y = np.random.randint(self.grid_size)
            self.grid[x, y] = self.treasure_reward
            self.treasure_pos.append((x, y))

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.agent_pos = (0, 0)
        self.generate_obstacles()
        self.generate_treasures()
        return self.grid, self.agent_pos

    def step(self, action):
        done = False
        reward = 0
        if action == 'up':
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 'down':
            if self.agent_pos[0] < self.grid_size - 1:
                self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 'left':
            if self.agent_pos[1] > 0:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == 'right':
            if self.agent_pos[1] < self.grid_size - 1:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        if self.agent_pos in self.obstacle_pos:
            done = True
            reward = -1
        elif self.agent_pos in self.treasure_pos:
            self.treasure_pos.remove(self.agent_pos)
            reward = self.treasure_reward
            if len(self.treasure_pos) == 0:
                done = True
        return self.grid, self.agent_pos, reward, done

    def render(self):
        fig, ax = plt.subplots()
        ax.imshow(self.grid, cmap='coolwarm')
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.grid(color='w', linewidth=2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for obs_pos in self.obstacle_pos:
            rect = plt.Rectangle(obs_pos, 1, 1, color='red')
            ax.add_patch(rect)
        for treasure_pos in self.treasure_pos:
            rect = plt.Rectangle(treasure_pos, 1, 1, color='green')
            ax.add_patch(rect)
        rect = plt.Rectangle(self.agent_pos, 1, 1, color='blue')
        ax.add_patch(rect)
        return fig


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, grid, agent_pos):
        return np.random.choice(self.action_space)


# Creating the App
env = TreasureHuntEnv()
agent = RandomAgent(env.action_space)

# Define a function to run the simulation for the specified number of steps


def run_simulation(env, agent, num_steps):
    # Render the initial state of the grid
    fig = env.render()
    st.pyplot(fig)
    # Loop through the specified number of steps
    for i in range(num_steps):
        # Choose an action based on the current state of the grid
        action = agent.act(env.grid, env.agent_pos)
        # Take the chosen action and get the new state of the grid and the reward
        grid, agent_pos, reward, done = env.step(action)
        # Render the updated state of the grid
        fig = env.render()
        st.pyplot(fig)
        # If the game is over, break out of the loop
        if done:
            break

# Define the main function for the Streamlit app
def main():
    # Set the page title
    st.set_page_config(page_title='Treasure Hunt')
    # Set the page header
    st.title('Treasure Hunt')
    # Set the default agent type and number of steps
    agent_type = st.sidebar.selectbox('Select Agent', ['Random'])
    num_steps = st.sidebar.slider(
        'Number of Steps', min_value=1, max_value=100, value=10, step=1)
    # Run the simulation
    if agent_type == 'Random':
        run_simulation(env, agent, num_steps)


if __name__ == '__main__':
    main()
