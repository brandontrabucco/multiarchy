"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.envs.normalized_env import NormalizedEnv
from multiarchy.distributions.tanh_gaussian import TanhGaussian
from multiarchy.distributions.gaussian import Gaussian
from multiarchy.networks import dense

from multiarchy.agents.hierarchy_agent import HierarchyAgent
from multiarchy.agents.multi_agent import MultiAgent
from multiarchy.agents.policy_agent import PolicyAgent

from multiarchy.replay_buffers.step_replay_buffer import StepReplayBuffer
from multiarchy.loggers.tensorboard_logger import TensorboardLogger
from multiarchy.samplers.sampler import Sampler
from multiarchy.algorithms.sac import SAC

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
import ray


def run_experiment(
        env_class,
        env_kwargs=None,
        observation_key="observation",
):
    # run an experiment with multiple agents
    if env_kwargs is None:
        env_kwargs = {}

    # initialize the cluster of cpu cores
    ray.init()

    # a function that creates the training environment
    def create_env():
        return NormalizedEnv(env_class, **env_kwargs)

    # create a replay buffer to store data
    replay_buffer = StepReplayBuffer(max_num_steps=1000000)

    # create a logging instance
    logger = TensorboardLogger(replay_buffer, "./")

    # create policies for each level in the hierarchy
    policy = TanhGaussian(dense(observation_dim, action_dim * 2), std=None)
    qf1 = TanhGaussian(dense(observation_dim, 1), std=1.0)
    qf2 = TanhGaussian(dense(observation_dim, 1), std=1.0)
    target_qf1 = TanhGaussian(dense(observation_dim, 1), std=1.0)
    target_qf2 = TanhGaussian(dense(observation_dim, 1), std=1.0)

    # train the agent using soft actor critic
    algorithm = SAC(
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        replay_buffer,
        reward_scale=1.0,
        discount=0.99,
        initial_alpha=0.1,
        alpha_optimizer_kwargs=dict(lr=0.0003),
        target_entropy=(-action_dim),
        logger=logger)

    # create a single agent to manage the hierarchy
    agent = PolicyAgent(policy, algorithm=algorithm)

    # create an agent to interact with an environment
    def create_agent():

        # create policies for each level in the hierarchy
        policy = TanhGaussian(dense(observation_dim, action_dim * 2))

        # create a single agent to manage the hierarchy
        return PolicyAgent(policy)

    # make a sampler to collect data to train the hierarchy
    sampler = Sampler(
        create_env,
        create_agent,
        max_path_length=1000,
        logger=logger,
        logging_prefix="sampler/")





if __name__ == "__main__":

    # initialize the cluster of cpu cores
    ray.init()

    # a function that creates the training environment
    def create_env():
        return NormalizedEnv(HumanoidEnv)

    # initialize the environment to track the cardinality of actions
    env = create_env()
    action_dim = env.action_space.low.size
    observation_dim = env.observation_space.spaces["observation"].low.size

    # create an agent to interact with an environment
    def create_agent():

        # create policies for each level in the hierarchy
        top_policy = Gaussian(dense(observation_dim, observation_dim))
        mid_policy = Gaussian(dense(observation_dim * 2, observation_dim))
        low_policy = Gaussian(dense(observation_dim * 2, action_dim))

        # create a single agent to manage the hierarchy
        return HierarchyAgent(
            MultiAgent(PolicyAgent(top_policy), time_skip=25),
            MultiAgent(PolicyAgent(mid_policy), time_skip=5),
            PolicyAgent(low_policy, time_skip=1))

    # create an agent in the main thread
    agent = create_agent()

    # create a replay buffer to store data
    replay_buffer = StepReplayBuffer(max_num_steps=1000000)

    # create a logging instance
    logger = TensorboardLogger(replay_buffer, "./")

    # make a sampler to collect data to train the hierarchy
    sampler = Sampler(
        create_env,
        create_agent,
        max_path_length=1000,
        logger=logger,
        logging_prefix="sampler/")

    # collect some data
    paths, num_steps = sampler.collect(
        10000,
        deterministic=False,
        render=True)

    sampler.set_weights(agent.get_weights())

    # insert the data into a replay buffer
    for path in paths:
        replay_buffer.insert_path(*path)
