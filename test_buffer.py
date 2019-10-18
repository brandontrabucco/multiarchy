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
        max_num_steps=1000000)

    replay_buffer.insert_path(
        [{"observation": np.array(1), "goal": np.array(1)},
         {"observation": np.array(2), "goal": np.array(2)},
         {"observation": np.array(3), "goal": np.array(3)},
         {"observation": np.array(4), "goal": np.array(4)}],
        [np.array(-1), np.array(-2), np.array(-3), np.array(-4)],
        [np.array(-.1), np.array(-.2), np.array(-.3), np.array(-.4)])

    obs, act, rew, next_obs, term = replay_buffer.sample(8)

    import ipdb; ipdb.set_trace()
