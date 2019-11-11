"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy import maybe_initialize_process
from multiarchy.envs.normalized_env import NormalizedEnv
from multiarchy.distributions.gaussian import Gaussian
from multiarchy.networks import dense
from multiarchy.agents.policy_agent import PolicyAgent
from multiarchy.replay_buffers.step_replay_buffer import StepReplayBuffer
from multiarchy.loggers.tensorboard_logger import TensorboardLogger
from multiarchy.samplers.parallel_sampler import ParallelSampler
from multiarchy.algorithms.ddpg import DDPG
import numpy as np
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


if __name__ == "__main__":

    # initialize tensorflow and the multiprocessing interface
    maybe_initialize_process()

    # initialize the environment to track the cardinality of actions
    env = NormalizedEnv(HalfCheetahEnv)
    action_dim = env.action_space.low.size
    observation_dim = env.observation_space.spaces[
        "observation"].low.size

    # create a replay buffer to store data
    replay_buffer = StepReplayBuffer(
        max_num_steps=10000)

    # create a logging instance
    logger = TensorboardLogger(
        replay_buffer, "./")

    # create policies for each level in the hierarchy
    policy = Gaussian(
        dense(
            observation_dim,
            action_dim,
            hidden_size=400,
            num_hidden_layers=2,
            output_activation="tanh"),
        optimizer_kwargs=dict(learning_rate=0.005),
        tau=0.005,
        std=0.1)
    target_policy = policy.clone()

    qf = Gaussian(
        dense(
            observation_dim + action_dim,
            1,
            hidden_size=400,
            num_hidden_layers=2),
        optimizer_kwargs=dict(learning_rate=0.005),
        tau=0.005,
        std=1.0)
    target_qf = qf.clone()

    # create a single agent to manage the hierarchy
    agent = PolicyAgent(
        policy,
        algorithm=algorithm,
        observation_key="observation")

    # make a sampler to collect data to warm up the hierarchy
    sampler = ParallelSampler(
        env,
        agent,
        max_path_length=100,
        num_workers=2)

    # collect more training samples
    sampler.set_weights(agent.get_weights())
    paths, returns, num_steps = sampler.collect(
        variant["num_warm_up_steps"],
        deterministic=False,
        keep_data=True,
        workers_to_use=variant["num_workers"])

    # insert the samples into the replay buffer
    for o, a, r in paths:
        replay_buffer.insert_path(o, a, r)

