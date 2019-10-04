"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.samplers.remote.sampler import RemoteSampler
import ray


class Sampler(object):

    def __init__(
            self,
            create_env,
            create_agent,
            max_path_length=1000,
            logger=None,
            logging_prefix="sampler/"
    ):
        # parameters to control sampling from the environment.
        self.sampler = RemoteSampler.remote(
            create_env,
            create_agent,
            max_path_length=max_path_length,
            logger=logger,
            logging_prefix=logging_prefix)

    def set_weights(
            self,
            weights
    ):
        # set the weights for the agent in this sampler
        self.sampler.set_weights.remote(weights)

    def collect(
            self,
            min_num_steps_to_collect,
            deterministic=False,
            render=False,
            render_kwargs=None
    ):
        # collect num_episodes amount of paths and track various things
        return ray.get(self.sampler.collect.remote(
            min_num_steps_to_collect,
            deterministic=deterministic,
            render=render,
            render_kwargs=render_kwargs))
