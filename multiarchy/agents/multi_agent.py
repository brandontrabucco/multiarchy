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
            self.goal = [goal for _agent in self.agents]
            for agent in self.agents:
                self.action.append(agent.react(
                    observation,
                    time_step,
                    goal,
                    deterministic=deterministic))

        # return the latest action and goal that was sampled by the agent
        return self.action, self.action, self.goal
