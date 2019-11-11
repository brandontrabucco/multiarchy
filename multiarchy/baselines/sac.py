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


sac_variant = dict(
    max_num_steps=1000000,
    logging_dir="./",
    hidden_size=400,
    num_hidden_layers=2,
    reward_scale=1.0,
    discount=0.99,
    initial_alpha=0.1,
    policy_learning_rate=0.0003,
    qf_learning_rate=0.0003,
    tau=0.005,
    batch_size=256,
    max_path_length=1000,
    num_workers=2,
    num_warm_up_steps=10000,
    num_steps_per_epoch=1000,
    num_steps_per_eval=10000,
    num_epochs_per_eval=1,
    num_epochs=10000)


def sac(
        variant,
        env_class,
        env_kwargs=None,
        observation_key="observation",
):
    # initialize tensorflow and the multiprocessing interface
    maybe_initialize_process()

    # run an experiment with multiple agents
    if env_kwargs is None:
        env_kwargs = {}

    # initialize the environment to track the cardinality of actions
    env = NormalizedEnv(env_class, **env_kwargs)
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
        dense(
            observation_dim,
            action_dim * 2,
            hidden_size=variant["hidden_size"],
            num_hidden_layers=variant["num_hidden_layers"]),
        optimizer_kwargs=dict(learning_rate=variant["policy_learning_rate"]),
        tau=variant["tau"],
        std=None)

    # create critics for each level in the hierarchy
    qf1 = Gaussian(
        dense(
            observation_dim + action_dim,
            1,
            hidden_size=variant["hidden_size"],
            num_hidden_layers=variant["qf_learning_rate"]),
        optimizer_kwargs=dict(learning_rate=variant["qf_learning_rate"]),
        tau=variant["tau"],
        std=1.0)
    target_qf1 = qf1.clone()

    # create critics for each level in the hierarchy
    qf2 = Gaussian(
        dense(
            observation_dim + action_dim,
            1,
            hidden_size=variant["hidden_size"],
            num_hidden_layers=variant["num_hidden_layers"]),
        optimizer_kwargs=dict(learning_rate=variant["qf_learning_rate"]),
        tau=variant["tau"],
        std=1.0)
    target_qf2 = qf2.clone()

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
        alpha_optimizer_kwargs=dict(learning_rate=variant["policy_learning_rate"]),
        target_entropy=(-action_dim),
        observation_key=observation_key,
        batch_size=variant["batch_size"],
        logger=logger,
        logging_prefix="sac/")

    # create a single agent to manage the hierarchy
    agent = PolicyAgent(
        policy,
        algorithm=algorithm,
        observation_key=observation_key)

    # make a sampler to collect data to warm up the hierarchy
    sampler = ParallelSampler(
        env,
        agent,
        max_path_length=variant["max_path_length"],
        num_workers=variant["num_workers"])

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

    #  train for a specified number of iterations
    for iteration in range(variant["num_epochs"]):

        if iteration % variant["num_epochs_per_eval"] == 0:

            # evaluate the policy at this step
            sampler.set_weights(agent.get_weights())
            paths, eval_returns, num_steps = sampler.collect(
                variant["num_steps_per_eval"],
                deterministic=True,
                keep_data=False,
                workers_to_use=variant["num_workers"])
            logger.record("eval_mean_return", np.mean(eval_returns))

        # collect more training samples
        sampler.set_weights(agent.get_weights())
        paths, train_returns, num_steps = sampler.collect(
            variant["num_steps_per_epoch"],
            deterministic=False,
            keep_data=True,
            workers_to_use=1)
        logger.record("train_mean_return", np.mean(train_returns))

        # insert the samples into the replay buffer
        for o, a, r in paths:
            replay_buffer.insert_path(o, a, r)

        # train once each for the number of steps collected
        for i in range(num_steps):
            agent.train()
