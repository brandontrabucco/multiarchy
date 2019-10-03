"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(
            self,
            time_skip=1
    ):
        # a single agent in a graph of many agents
        self.time_skip = time_skip
        self.action = None
        self.goal = None

    @abstractmethod
    def react(
            self,
            observation,
            time_step,
            goal,
            deterministic=False
    ):
        # return atomic actions, hierarchy of actions, and hierarchy of goals
        return NotImplemented
