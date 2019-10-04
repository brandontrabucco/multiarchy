"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy import nested_apply
from multiarchy.replay_buffers.replay_buffer import ReplayBuffer
import numpy as np
import ray


@ray.remote
class RemoteStepReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            max_num_steps=1000000
    ):
        ReplayBuffer.__init__(self)

        # parameters to control how the buffer is created and managed
        self.max_num_steps = max_num_steps

    def inflate_backend(
            self,
            x
    ):
        # create numpy arrays to store samples
        x = x if isinstance(x, np.ndarray) else np.array(x)
        return np.zeros_like(x, shape=[self.max_num_steps, *x.shape])

    def insert_backend(
            self,
            structure,
            data
    ):
        # insert samples into the numpy array
        structure[self.head, ...] = data

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # insert a path into the replay buffer
        self.total_paths += 1
        observations = observations[:self.max_num_steps]
        actions = actions[:self.max_num_steps]
        rewards = rewards[:self.max_num_steps]

        # inflate the replay buffer if not inflated
        if any([self.observations is None, self.actions is None, self.rewards is None,
                self.terminals is None]):
            self.observations = nested_apply(self.inflate_backend, observations[0])
            self.actions = nested_apply(self.inflate_backend, actions[0])
            self.rewards = self.inflate_backend(rewards[0])
            self.terminals = self.inflate_backend(rewards[0])

        # insert all samples into the buffer
        for time_step, (o, a, r) in enumerate(zip(observations, actions, rewards)):
            nested_apply(self.insert_backend, self.observations, o)
            nested_apply(self.insert_backend, self.actions, a)
            self.insert_backend(self.rewards, r)
            self.insert_backend(self.terminals, time_step)

            # increment the head and size
            self.head = (self.head + 1) % self.max_num_steps
            self.size = min(self.size + 1, self.max_num_steps)
            self.total_steps += 1

    def sample(
            self,
            batch_size,
            time_skip=1,
            hierarchy_selector=(lambda x: x)
    ):
        # sample transition for a hierarchy of policies
        idx = np.random.choice(
            self.size, size=batch_size, replace=(self.size < batch_size))
        idx = idx - self.terminals[idx, ...].astype(np.int32) % time_skip
        next_idx = (idx + time_skip + 1) % self.max_num_steps
        intermediate_ids = [(idx + i) % self.max_num_steps for i in range(time_skip)]

        def inner_sample(data):
            return data[idx, ...]

        def inner_sample_last(data):
            return data[next_idx, ...]

        # sample current batch from a nested structure
        observations = nested_apply(inner_sample, self.observations)
        observations["goal"] = hierarchy_selector(observations["goal"])
        actions = hierarchy_selector(nested_apply(inner_sample, self.actions))

        # sum the rewards across the horizon where valid
        rewards = 0.0
        for j in intermediate_ids:
            rewards = rewards + (self.rewards[j, ...] * np.greater_equal(
                self.terminals[j, ...], self.terminals[idx, ...]).astype(np.float32))

        # sample current batch from a nested structure
        next_observations = nested_apply(inner_sample_last, self.observations)
        next_observations["goal"] = hierarchy_selector(next_observations["goal"])
        terminals = np.greater_equal(
            inner_sample_last(self.terminals),
            inner_sample(self.terminals)).astype(np.float32)

        # return the samples in a batch
        return observations, actions, rewards, next_observations, terminals
