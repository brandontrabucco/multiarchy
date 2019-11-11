"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy import maybe_initialize_process
from multiarchy.envs.normalized_env import NormalizedEnv
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

    # create a replay buffer to store data
    replay_buffer = StepReplayBuffer(
        max_num_steps=1000)

    observations = []
    actions = []
    rewards = []

    for j in range(10):

        for i in range(1000):

            observation = {"observation": np.array(i), "goal": np.array(i)}
            action = np.array(-i)
            reward = np.array(-i / 10)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

        replay_buffer.insert_path(
            observations,
            actions,
            rewards)

    obs, act, rew, next_obs, term = replay_buffer.sample(8)

    import ipdb; ipdb.set_trace()
