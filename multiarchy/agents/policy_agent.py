"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.agents.agent import Agent
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
            self.goal = goal
            gs = goal if isinstance(goal, list) else [goal]
            inputs = np.concatenate([self.input_selector(observation), *gs], -1)
            self.action = (self.policy.expected_value(
                inputs) if deterministic else self.policy.sample(inputs))[0, ...]

        # return the latest action and goal that was sampled by the agent
        return self.action, self.action, self.goal
