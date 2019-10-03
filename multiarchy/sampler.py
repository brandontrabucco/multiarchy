"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import numpy as np
import ray


class Sampler(object):

    def __init__(
            self,
            create_env,
            max_path_length=1000,
            logger=None,
            logging_prefix="sampler/"
    ):
        # parameters to control sampling from the environment.
        self.env = create_env()
        self.max_path_length = max_path_length
        self.logger = logger
        self.logging_prefix = logging_prefix

    def collect(
            self,
            agent,
            min_num_steps_to_collect,
            deterministic=False,
            render=False,
            render_kwargs=None
    ):
        # collect num_episodes amount of paths and track various things
        if render_kwargs is None:
            render_kwargs = {}

        # store data to pass to the replay buffer
        paths = []
        all_returns = []

        # start collecting many trajectories
        num_steps_collected = 0
        while num_steps_collected < min_num_steps_to_collect:

            # keep track of observations actions and rewards
            observations = []
            actions = []
            rewards = []

            # reset the environment at the start of each trajectory
            observation_t = self.env.reset()
            path_return = 0.0

            # unroll the episode until done or max_path_length is attained
            for time_step in range(self.max_path_length):

                # check if the environment has a goal and send it in
                num_steps_collected += 1
                atoms_t, actions_t, goals_t = agent.react(
                    observation_t,
                    time_step,
                    observation_t["goal"] if "goal" in observation_t else [],
                    deterministic=deterministic)

                # save the observation and the actions from the agent
                observation_t["goal"] = goals_t
                observations.append(observation_t)
                actions.append(actions_t)

                # update the environment with the atomic actions
                observation_t, reward_t, done, info = self.env.step(atoms_t)
                rewards.append(reward_t)
                path_return += reward_t

                # and possibly render the updated environment (to a video)
                if render:
                    self.env.render(**render_kwargs)

                # exit if the simulation has reached a terminal state
                if done:
                    break

            # save the episode into a list to send to the replay buffer
            paths.append((observations, actions, rewards))
            all_returns.append(path_return)

        # log the average return achieved by the agent for these steps
        if self.logger is not None:
            self.logger.record.remote(
                self.logging_prefix + "return_mean", np.mean(all_returns))
            self.logger.record.remote(
                self.logging_prefix + "return_std", np.std(all_returns))


RemoteSampler = ray.remote(Sampler)
