"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.agents.agent import Agent


class HierarchyAgent(Agent):

    def __init__(
            self,
            *agents,
            time_skip=1
    ):
        Agent.__init__(self, time_skip=time_skip)

        # a struct to store key methods for a hierarchy level
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
            self.goals = []
            self.action = []
            for agent in self.agents:
                self.goals.append(goal)
                action = agent.react(
                    observation,
                    time_step,
                    goal,
                    deterministic=deterministic)
                self.action.append(action)

                # high level actions are low level goals
                goal = action

        # return the actions that were sampled by all the agents
        return self.action[-1], self.action, self.goals
