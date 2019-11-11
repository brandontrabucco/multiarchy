"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy import maybe_initialize_process
from multiarchy.envs.normalized_env import NormalizedEnv
from multiarchy.envs.counter_env import CounterEnv
from multiarchy.distributions.tanh_gaussian import TanhGaussian
from multiarchy.distributions.gaussian import Gaussian
from multiarchy.networks import dense
from multiarchy.agents.policy_agent import PolicyAgent
from multiarchy.replay_buffers.step_replay_buffer import StepReplayBuffer
from multiarchy.loggers.tensorboard_logger import TensorboardLogger
from multiarchy.samplers.parallel_sampler import ParallelSampler
from multiarchy.algorithms.sac import SAC
import numpy as np


if __name__ == "__main__":

    # initialize tensorflow and the multiprocessing interface
    maybe_initialize_process()

    # initialize the environment to track the cardinality of actions
    env = NormalizedEnv(CounterEnv, max_path_length=1000)
    action_dim = env.action_space.low.size
    observation_dim = env.observation_space.spaces[
        "observation"].low.size

    # create a replay buffer to store data
    replay_buffer = StepReplayBuffer(
        max_num_steps=10000)

    # create policies for each level in the hierarchy
    policy = Gaussian(
        dense(
            observation_dim,
            action_dim,
            hidden_size=32,
            num_hidden_layers=2,
            output_activation="tanh"),
        std=0.2)

    # create a single agent to manage the hierarchy
    agent = PolicyAgent(
        policy,
        observation_key="observation")

    # make a sampler to collect data to warm up the hierarchy
    sampler = ParallelSampler(
        env,
        agent,
        max_path_length=1000,
        num_workers=2)

    # collect more training samples
    sampler.set_weights(agent.get_weights())
    paths, returns, num_steps = sampler.collect(
        10000,
        deterministic=False,
        keep_data=True,
        workers_to_use=2)

    # insert the samples into the replay buffer
    for o, a, r in paths:
        replay_buffer.insert_path(o, a, r)

    #  train for a specified number of iterations
    for iteration in range(10):

        # collect more training samples
        sampler.set_weights(agent.get_weights())
        paths, train_returns, num_steps = sampler.collect(
            1000,
            deterministic=False,
            keep_data=True,
            workers_to_use=1)

        # insert the samples into the replay buffer
        for o, a, r in paths:
            replay_buffer.insert_path(o, a, r)

    import ipdb; ipdb.set_trace()
