import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


from Agent.QLearningAgent import QLearningAgent
from Agent.RandomAgent import RandomAgent
from TreasureHuntEnv.TreasureHuntEnv import TreasureHuntEnv


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
    if st.button(":running: Lets Go!",  type="secondary", disabled=False, use_container_width=False):

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
        if count > 1:
            st.balloons()
            st.success('We have reached our treasure!', icon="✅")


# Run the app
if __name__ == '__main__':
    app()
