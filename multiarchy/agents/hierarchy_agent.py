"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.agents.agent import Agent
import pickle as pkl


class HierarchyAgent(Agent):

    def __init__(
            self,
            *agents,
            time_skip=1
    ):
        Agent.__init__(self, time_skip=time_skip)

        # a struct to store key methods for a hierarchy level
        self.agents = agents

    def __getstate__(
            self
    ):
        # handle pickle actions so the agent can be sent between threads
        state = Agent.__getstate__(self)
        return dict(agents=[pkl.dumps(agent) for agent in self.agents], **state)

    def __setstate__(
            self,
            state
    ):
        # handle pickle actions so the agent can be sent between threads
        Agent.__setstate__(self, state)
        self.agents = [pkl.loads(x) for x in state["agents"]]

    def train(
            self,
            iteration=None,
            hierarchy_selector=(lambda x: x)
    ):
        # train the algorithm using this replay buffer
        Agent.train(self, iteration=iteration, hierarchy_selector=hierarchy_selector)
        for i, agent in enumerate(self.agents):
            agent.train(
                iteration=iteration,
                hierarchy_selector=(lambda x: hierarchy_selector(x)[i]))

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
            self.action = goal
            self.stack = []
            self.goal = []

            for agent in self.agents:
                result = agent.react(
                    observation,
                    time_step,
                    self.action,
                    deterministic=deterministic)

                self.action = result[0]
                self.stack.append(result[1])
                self.goal.append(result[2])

        # return the actions that were sampled by all the agents
        return self.action, self.stack, self.goal
