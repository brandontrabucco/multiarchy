"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.launch import launch_local
from multiarchy.baselines.ppo import ppo, ppo_variant
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


if __name__ == "__main__":

    # parameters for the learning experiment
    variant = dict(
        max_path_length=1000,
        max_num_paths=1000,
        logging_dir="half_cheetah/ppo/",
        hidden_size=400,
        num_hidden_layers=2,
        reward_scale=1.0,
        discount=0.99,
        epsilon=0.1,
        lamb=0.95,
        off_policy_updates=10,
        critic_updates=32,
        policy_learning_rate=0.0001,
        vf_learning_rate=0.001,
        exploration_noise_std=0.5,
        num_workers=10,
        num_steps_per_epoch=10000,
        num_steps_per_eval=10000,
        num_epochs_per_eval=1,
        num_epochs=1000)

    # make sure that all the right parameters are here
    assert all([x in variant.keys() for x in ppo_variant.keys()])

    # launch the experiment using ray
    launch_local(
        ppo,
        variant,
        HalfCheetahEnv,
        num_seeds=3)
