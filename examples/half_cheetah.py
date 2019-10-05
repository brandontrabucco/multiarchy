"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.sac import sac
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import ray


@ray.remote
def run_experiment(experiment_id):

    # change the logging directory and set parameters
    variant = dict(
        max_num_steps=1000000,
        logging_dir="half_cheetah/sac/{}/".format(experiment_id),
        reward_scale=1.0,
        discount=0.99,
        initial_alpha=0.1,
        lr=0.0003,
        tau=0.005,
        batch_size=256,
        max_path_length=1000,
        num_warm_up_steps=10000,
        num_steps_per_epoch=1000,
        num_steps_per_eval=10000,
        num_epochs_per_eval=10,
        num_epochs=10000)

    # run an experiment using these parameters
    return sac(variant, HalfCheetahEnv)


if __name__ == "__main__":

    # turn on the ray cluster
    ray.init()

    # run several experiments with the same parameters
    ray.get([run_experiment.remote(seed) for seed in range(1)])
