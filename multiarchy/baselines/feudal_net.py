"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.envs.normalized_env import NormalizedEnv
from multiarchy.distributions.gaussian import Gaussian
from multiarchy.networks import dense
from multiarchy.agents.policy_agent import PolicyAgent
from multiarchy.agents.hierarchy_agent import HierarchyAgent
from multiarchy.replay_buffers.step_replay_buffer import StepReplayBuffer
from multiarchy.loggers.tensorboard_logger import TensorboardLogger
from multiarchy.samplers.parallel_sampler import Sampler
from multiarchy.algorithms.td3 import TD3


feudal_net_variant = dict(
    max_num_steps=1000000,
    logging_dir="./",
    hidden_size=400,
    num_hidden_layers=2,
    num_levels=2,
    time_skip=5,
    exploration_noise_std=0.5,
    reward_scale=1.0,
    discount=0.99,
    target_clipping=0.5,
    target_noise=0.2,
    lr=0.0003,
    tau=0.005,
    batch_size=256,
    max_path_length=1000,
    num_warm_up_steps=10000,
    num_steps_per_epoch=1000,
    num_steps_per_eval=10000,
    num_epochs_per_eval=1,
    num_epochs=10000)


def feudal_net(
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

    # create a hierarchy of agents
    all_agents = []
    for level in range(variant["num_levels"]):

        # track the cardinality of this levels actions and observations
        input_dim = observation_dim * 2 if level < variant["num_levels"] - 1 else observation_dim
        output_dim = observation_dim if level < variant["num_levels"] - 1 else action_dim

        # create policies for each level in the hierarchy
        policy = Gaussian(
            dense(
                input_dim,
                output_dim,
                hidden_size=variant["hidden_size"],
                num_hidden_layers=variant["num_hidden_layers"]),
            optimizer_kwargs=dict(lr=variant["lr"]),
            tau=variant["tau"],
            std=variant["exploration_noise_std"])
        target_policy = Gaussian(
            dense(
                input_dim,
                output_dim,
                hidden_size=variant["hidden_size"],
                num_hidden_layers=variant["num_hidden_layers"]),
            optimizer_kwargs=dict(lr=variant["lr"]),
            tau=variant["tau"],
            std=variant["exploration_noise_std"])
        qf1 = Gaussian(
            dense(
                input_dim + output_dim,
                1,
                hidden_size=variant["hidden_size"],
                num_hidden_layers=variant["num_hidden_layers"]),
            optimizer_kwargs=dict(lr=variant["lr"]),
            tau=variant["tau"],
            std=1.0)
        qf2 = Gaussian(
            dense(
                input_dim + output_dim,
                1,
                hidden_size=variant["hidden_size"],
                num_hidden_layers=variant["num_hidden_layers"]),
            optimizer_kwargs=dict(lr=variant["lr"]),
            tau=variant["tau"],
            std=1.0)
        target_qf1 = Gaussian(
            dense(
                input_dim + output_dim,
                1,
                hidden_size=variant["hidden_size"],
                num_hidden_layers=variant["num_hidden_layers"]),
            optimizer_kwargs=dict(lr=variant["lr"]),
            tau=variant["tau"],
            std=1.0)
        target_qf2 = Gaussian(
            dense(
                input_dim + output_dim,
                1,
                hidden_size=variant["hidden_size"],
                num_hidden_layers=variant["num_hidden_layers"]),
            optimizer_kwargs=dict(lr=variant["lr"]),
            tau=variant["tau"],
            std=1.0)

        def observation_selector(x):
            return x[observation_key]

        # train the agent using soft actor critic
        algorithm = TD3(
            policy,
            target_policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            replay_buffer,
            reward_scale=variant["reward_scale"],
            discount=variant["discount"],
            target_clipping=variant["target_clipping"],
            target_noise=variant["target_noise"],
            input_selector=observation_selector,
            batch_size=variant["batch_size"],
            update_every=variant["time_skip"] ** (variant["num_levels"] - 1 - level),
            logger=logger,
            logging_prefix="td3_{}/".format(level))

        # create a single agent to manage the hierarchy
        all_agents.append(PolicyAgent(
            policy,
            time_skip=variant["time_skip"] ** (variant["num_levels"] - 1 - level),
            algorithm=algorithm,
            input_selector=observation_selector))

    # make the agents a hierarchy
    agent = HierarchyAgent(*all_agents)

    # create an agent to interact with an environment
    def create_agent():
        return HierarchyAgent(*[
            PolicyAgent(
                Gaussian(
                    dense(
                        observation_dim * 2 if level < variant["num_levels"] - 1 else observation_dim,
                        observation_dim if level < variant["num_levels"] - 1 else action_dim),
                    optimizer_kwargs=dict(lr=variant["lr"]),
                    tau=variant["tau"],
                    std=variant["exploration_noise_std"]),
                time_skip=variant["time_skip"] ** (variant["num_levels"] - 1 - inner_level),
                input_selector=observation_selector)
            for inner_level in range(variant["num_levels"])])

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
        keep_data=True)

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
                keep_data=False)

        # collect more training samples
        train_sampler.set_weights(agent.get_weights())
        paths, num_steps = train_sampler.collect(
            variant["num_steps_per_epoch"],
            deterministic=False,
            keep_data=True)

        # insert the samples into the replay buffer
        for o, a, r in paths:
            replay_buffer.insert_path(o, a, r)

        # train once each for the number of steps collected
        for i in range(num_steps):
            agent.train()
