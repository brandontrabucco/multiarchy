"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(
            self,
            time_skip=1,
            goal_skip=1,
            algorithm=None
    ):
        # a single agent in a graph of many agents
        self.time_skip = time_skip
        self.goal_skip = goal_skip
        self.iteration = 0
        self.algorithm = algorithm
        self.action = None
        self.stack = None
        self.goal = None

    def __getstate__(
            self
    ):
        # handle pickle actions so the agent can be sent between threads
        return dict(iteration=self.iteration, time_skip=self.time_skip)

    def __setstate__(
            self,
            state
    ):
        # handle pickle actions so the agent can be sent between threads
        self.iteration = state["iteration"]
        self.time_skip = state["time_skip"]

    def train(
            self,
            iteration=None,
            hierarchy_selector=(lambda x: x)
    ):
        # train the algorithm using this replay buffer
        self.iteration = self.iteration + 1 if iteration is None else iteration
        if self.algorithm is not None:
            self.algorithm.fit(
                self.iteration,
                time_skip=self.time_skip,
                goal_skip=self.goal_skip,
                hierarchy_selector=hierarchy_selector)

    @abstractmethod
    def get_weights(
            self,
    ):
        # return a nested structure of weights for the hierarchy
        return NotImplemented

    @abstractmethod
    def set_weights(
            self,
            weights
    ):
        # assign a nested structure of weights for the hierarchy
        return NotImplemented

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
