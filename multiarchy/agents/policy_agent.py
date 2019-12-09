"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.agents.agent import Agent
from multiarchy import flatten
import numpy as np


class PolicyAgent(Agent):

    def __init__(
            self,
            policy,
            observation_key="observation",
            time_skip=1,
            goal_skip=1,
            algorithm=None
    ):
        Agent.__init__(self, goal_skip=goal_skip, time_skip=time_skip, algorithm=algorithm)

        # story the policy and a selector into the observation dictionary
        self.policy = policy
        self.observation_key = observation_key

    def __getstate__(
            self
    ):
        # handle pickle actions so the agent can be sent between threads
        state = Agent.__getstate__(self)
        return dict(
            observation_key=self.observation_key,
            policy=self.policy, **state)

    def __setstate__(
            self,
            state
    ):
        # handle pickle actions so the agent can be sent between threads
        Agent.__setstate__(self, state)
        self.observation_key = state["observation_key"]
        self.policy = state["policy"]

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
            inputs = np.concatenate([observation[self.observation_key], *self.goal], -1)[None, ...]
            self.stack = self.action = (self.policy.expected_value(
                inputs) if deterministic else self.policy.sample(inputs))[0][0, ...].numpy()

        # return the latest action and goal that was sampled by the agent
        return self.action, self.stack, self.goal
