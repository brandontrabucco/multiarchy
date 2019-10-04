"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.sac import sac, sac_variant
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from copy import deepcopy
import ray


def run_experiment(experiment_id):

    # change the logging directory and set parameters
    variant = deepcopy(sac_variant)
    variant = dict(
        max_num_steps=1000000,
        logging_dir="half_cheetah/sac/{}/".format(experiment_id),
        reward_scale=1.0,
        discount=0.99,
        initial_alpha=0.1,
        lr=0.0003,
        max_path_length=1000,
        num_warm_up_steps=10000,
        num_steps_per_epoch=1000,
        num_steps_per_eval=10000,
        num_epochs_per_eval=1,
        num_epochs=10000)

    # run an experiment using these parameters
    return sac(variant, HalfCheetahEnv)


if __name__ == "__main__":

    # turn on the ray cluster
    ray.init()

    # run several experiments with the same parameters
    num_seeds = 1
    run_experiment(0)
