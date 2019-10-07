"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from copy import deepcopy
import ray
import tensorflow as tf


def launch(
    baseline,
    variant,
    env_class,
    env_kwargs=None,
    observation_key="observation",
    num_cpus=6,
    num_gpus=1,
    num_seeds=2,
):
    # starts the ray cluster and launches several experiments on it
    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        redis_max_memory=1024 * 1024 * 1024,
        object_store_memory=1024 * 1024 * 1024)

    # wrap the baseline in order to share resources
    @ray.remote(
        num_cpus=num_cpus // num_seeds,
        num_gpus=num_gpus / num_seeds)
    def baseline_remote(
        seed_variant,
        seed_env_class,
        seed_env_kwargs=None,
        seed_observation_key="observation",
    ):
        # prevent any process from consuming all gpu memory
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # start training the baseline
        return baseline(
            seed_variant,
            seed_env_class,
            env_kwargs=seed_env_kwargs,
            observation_key=seed_observation_key)

    # launch the experiments on the ray cluster
    results = []
    for seed in range(2):
        seed_variant = deepcopy(variant)
        seed_variant["logging_dir"] += "{}/".format(seed)
        results.append(
            baseline_remote.remote(
                seed_variant,
                env_class,
                seed_env_kwargs=env_kwargs,
                seed_observation_key=observation_key))

    # wait for every experiment to finish
    ray.get(results)
