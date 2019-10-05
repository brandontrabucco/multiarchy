"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.envs.normalized_env import NormalizedEnv
from multiarchy.distributions.tanh_gaussian import TanhGaussian
from multiarchy.distributions.gaussian import Gaussian
from multiarchy.networks import dense
from multiarchy.agents.policy_agent import PolicyAgent
from multiarchy.replay_buffers.step_replay_buffer import StepReplayBuffer
from multiarchy.loggers.tensorboard_logger import TensorboardLogger
from multiarchy.samplers.sampler import Sampler
from multiarchy.algorithms.sac import SAC
import ray


sac_variant = dict(
    max_num_steps=1000000,
    logging_dir="./",
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


# TODO: the core dump is most likely due to an error in hierarchy_selector


def sac(
        variant,
        env_class,
        env_kwargs=None,
        observation_key="observation",
):
    # run an experiment with multiple agents
    if env_kwargs is None:
        env_kwargs = {}

    # a function that creates the training environment
    def create_env():
        return NormalizedEnv(env_class, **env_kwargs)

    # initialize the environment to track the cardinality of actions
    env = create_env()
    action_dim = env.action_space.low.size
    observation_dim = env.observation_space.spaces[
        observation_key].low.size

    # create a replay buffer to store data
    replay_buffer = StepReplayBuffer(
        max_num_steps=variant["max_num_steps"])

    # create a logging instance
    logger = TensorboardLogger(
        replay_buffer, variant["logging_dir"])

    # create policies for each level in the hierarchy
    policy = TanhGaussian(
        dense(observation_dim, action_dim * 2),
        optimizer_kwargs=dict(lr=variant["lr"]),
        std=None)
    qf1 = Gaussian(
        dense(observation_dim + action_dim, 1),
        optimizer_kwargs=dict(lr=variant["lr"]),
        std=1.0)
    qf2 = Gaussian(
        dense(observation_dim + action_dim, 1),
        optimizer_kwargs=dict(lr=variant["lr"]),
        std=1.0)
    target_qf1 = Gaussian(
        dense(observation_dim + action_dim, 1),
        optimizer_kwargs=dict(lr=variant["lr"]),
        std=1.0)
    target_qf2 = Gaussian(
        dense(observation_dim + action_dim, 1),
        optimizer_kwargs=dict(lr=variant["lr"]),
        std=1.0)

    def observation_selector(x):
        return x[observation_key]

    # train the agent using soft actor critic
    algorithm = SAC(
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        replay_buffer,
        reward_scale=variant["reward_scale"],
        discount=variant["discount"],
        initial_alpha=variant["initial_alpha"],
        alpha_optimizer_kwargs=dict(lr=variant["lr"]),
        target_entropy=(-action_dim),
        input_selector=observation_selector,
        logger=logger,
        logging_prefix="sac/")

    # create a single agent to manage the hierarchy
    agent = PolicyAgent(
        policy,
        algorithm=algorithm,
        input_selector=observation_selector)

    # create an agent to interact with an environment
    def create_agent():
        return PolicyAgent(
            TanhGaussian(
                dense(observation_dim, action_dim * 2), std=None),
            input_selector=observation_selector)

    # make a sampler to collect data to warm up the hierarchy
    warm_up_sampler = Sampler(
        create_env,
        create_agent,
        max_path_length=variant["max_path_length"],
        num_processes=(variant["num_warm_up_steps"] //
                       variant["max_path_length"]),
        logger=logger,
        logging_prefix="warm_up_sampler/")

    # make a sampler to collect data to train the hierarchy
    train_sampler = Sampler(
        create_env,
        create_agent,
        max_path_length=variant["max_path_length"],
        num_processes=(variant["num_steps_per_epoch"] //
                       variant["max_path_length"]),
        logger=logger,
        logging_prefix="train_sampler/")

    # make a sampler to collect data to evaluate the hierarchy
    eval_sampler = Sampler(
        create_env,
        create_agent,
        max_path_length=variant["max_path_length"],
        num_processes=(variant["num_steps_per_eval"] //
                       variant["max_path_length"]),
        logger=logger,
        logging_prefix="eval_sampler/")

    # collect more training samples
    warm_up_sampler.set_weights(agent.get_weights())
    paths, num_steps = warm_up_sampler.collect(
        variant["num_warm_up_steps"],
        deterministic=False,
        save_data=True)

    # insert the samples into the replay buffer
    for o, a, r in paths:
        replay_buffer.insert_path(o, a, r)

    #  train for a specified number of iterations
    for iteration in range(variant["num_epochs"]):

        if iteration % variant["num_epochs_per_eval"] == 0:

            # evaluate the policy at this step
            eval_sampler.set_weights(agent.get_weights())
            eval_sampler.collect(
                variant["num_steps_per_eval"],
                deterministic=True,
                save_data=False)

        # collect more training samples
        train_sampler.set_weights(agent.get_weights())
        paths, num_steps = train_sampler.collect(
            variant["num_steps_per_epoch"],
            deterministic=False,
            save_data=True)

        # insert the samples into the replay buffer
        for o, a, r in paths:
            replay_buffer.insert_path(o, a, r)

        # train once each for the number of steps collected
        for i in range(num_steps):
            agent.train(iteration)
