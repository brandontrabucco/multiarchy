"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.agents.agent import Agent
from multiarchy import flatten
import numpy as np


class PolicyAgent(Agent):

    def __init__(
            self,
            policy,
            time_skip=1,
            input_selector=(lambda x: x["observation"]),
    ):
        Agent.__init__(self, time_skip=time_skip)

        # story the policy and a selector into the observation dictionary
        self.policy = policy
        self.input_selector = input_selector

    def get_weights(
            self,
    ):
        # return a nested structure of weights for the hierarchy
        return self.policy.get_weights()

    def set_weights(
            self,
            weights
    ):
        # assign a nested structure of weights for the hierarchy
        self.policy.set_weights(weights)

    def react(
            self,
            observation,
            time_step,
            goal,
            deterministic=False,
    ):
        # determine if the current cell in the hierarchy is active
        if time_step % self.time_skip == 0:

            # choose to use the stochastic or deterministic policy
            self.goal = flatten(goal)
            inputs = np.concatenate([self.input_selector(observation), *self.goal], -1)[None, ...]
            self.action = (self.policy.expected_value(
                inputs) if deterministic else self.policy.sample(inputs))[0][0, ...].numpy()

        # return the latest action and goal that was sampled by the agent
        return self.action, self.action, self.goal
