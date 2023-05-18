
import random


class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return random.choice(self.action_space)

    def reset(self):
        pass
