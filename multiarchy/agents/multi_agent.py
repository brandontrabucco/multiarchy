"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.agents.agent import Agent


class MultiAgent(Agent):

    def __init__(
            self,
            *agents,
            time_skip=1
    ):
        Agent.__init__(self, time_skip=time_skip)

        # a list of several parallel agents
        self.agents = agents

    def get_weights(
            self,
    ):
        # return a nested structure of weights for the hierarchy
        return [agent.get_weights() for agent in self.agents]

    def set_weights(
            self,
            weights
    ):
        # assign a nested structure of weights for the hierarchy
        for w, agent in zip(weights, self.agents):
            agent.set_weights(w)

    def react(
            self,
            observation,
            time_step,
            goal,
            deterministic=False
    ):
        # determine if the current cell in the hierarchy is active
        if time_step % self.time_skip == 0:

            # get actions from the set of agents
            self.action = []
            self.stack = []
            self.goal = []

            for agent in self.agents:
                result = agent.react(
                    observation,
                    time_step,
                    goal,
                    deterministic=deterministic)

                self.action.append(result[0])
                self.stack.append(result[1])
                self.goal.append(result[2])

        # return the latest action and goal that was sampled by the agent
        return self.action, self.action, self.goal
