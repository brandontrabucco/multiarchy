"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.samplers.remote.sampler import RemoteSampler
import ray


class Sampler(object):

    def __init__(
            self,
            create_env,
            create_agent,
            max_path_length=1000,
            num_processes=1,
            logger=None,
            logging_prefix="sampler/"
    ):
        # parameters to control sampling from the environment.
        self.samplers = []
        self.num_processes = num_processes
        for i in range(num_processes):
            self.samplers.append(RemoteSampler.remote(
                create_env,
                create_agent,
                max_path_length=max_path_length,
                logger=logger,
                logging_prefix=logging_prefix[:(-1)] + "_{}/".format(i)))

    def set_weights(
            self,
            weights
    ):
        # set the weights for the agent in this sampler
        for sampler in self.samplers:
            sampler.set_weights.remote(weights)

    def collect(
            self,
            min_num_steps_to_collect,
            deterministic=False,
            save_data=False,
            render=False,
            render_kwargs=None
    ):
        # collect num_episodes amount of paths and track various things
        results = []
        for i, sampler in enumerate(self.samplers):
            target_size = (min_num_steps_to_collect // self.num_processes)
            if (min_num_steps_to_collect % self.num_processes) - i > 0:
                target_size += 1
            results.append(sampler.collect.remote(
                target_size,
                deterministic=deterministic,
                save_data=save_data,
                render=render,
                render_kwargs=render_kwargs))
        results = ray.get(results)
        paths = [path for item in results for path in item[0]]
        return paths, sum([item[1] for item in results])
