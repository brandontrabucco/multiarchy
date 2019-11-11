"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from gym.spaces import Box, Dict
from gym import Env
import numpy as np


class CounterEnv(Env):

    def render(
            self,
            mode='human'
    ):
        print(self.position)

    def __init__(
            self,
    ):
        self.observation_space = Dict({
            "observation": Box(np.array([-np.inf]), np.array([np.inf]))})
        self.action_space = Box(np.array([-1.0]), np.array([1.0]))
        self.position = np.array([0.0])

    def reset(
            self,
            **kwargs
    ):
        self.position = np.array([0.0])
        return {"observation": self.position}

    def step(
            self,
            action
    ):
        self.position = self.position + 1.0
        reward = 1.0
        return ({"observation": self.position},
                reward, False, {})
