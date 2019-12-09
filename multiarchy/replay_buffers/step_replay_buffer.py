"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy import nested_apply
from multiarchy.replay_buffers.replay_buffer import ReplayBuffer
import numpy as np


class StepReplayBuffer(ReplayBuffer):

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
            self.rewards = self.inflate_backend(np.squeeze(rewards[0]))
            self.terminals = self.inflate_backend(np.array([0, 0]))

        # insert all samples into the buffer
        for time_step, (o, a, r) in enumerate(zip(observations, actions, rewards)):
            nested_apply(self.insert_backend, self.observations, o)
            nested_apply(self.insert_backend, self.actions, a)
            self.insert_backend(self.rewards, np.squeeze(r))
            self.insert_backend(self.terminals, np.array([time_step, self.total_paths]))

            # increment the head and size
            self.head = (self.head + 1) % self.max_num_steps
            self.size = min(self.size + 1, self.max_num_steps)
            self.total_steps += 1

    def sample(
            self,
            batch_size,
            time_skip=1,
            goal_skip=1,
            hierarchy_selector=(lambda x: x)
    ):
        # handle cases when we want to sample everything
        batch_size = batch_size if batch_size > 0 else self.size

        # sample transition for a hierarchy of policies
        idx = np.random.choice(
            self.size, size=batch_size, replace=(self.size < batch_size))

        # force the samples to occur every time_skip
        idx = idx - self.terminals[idx, 0].astype(np.int32) % time_skip
        next_idx = np.minimum(idx + time_skip, self.max_num_steps)

        def sample_observations(data):
            return data[idx, ...]

        def sample_observations_last(data):
            return data[next_idx, ...]

        # sample current batch from a nested structure
        observations = nested_apply(sample_observations, self.observations)
        observations["goal"] = hierarchy_selector(observations["goal"])
        actions = hierarchy_selector(nested_apply(sample_observations, self.actions))

        # sum the rewards across the horizon where valid
        rewards = 0.0
        for j in [(idx + i) % self.max_num_steps for i in range(time_skip)]:
            rewards = rewards + (self.rewards[j, ...] * np.equal(
                self.terminals[j, 1], self.terminals[idx, 1]).astype(np.float32))

        # sample current batch from a nested structure
        next_observations = nested_apply(sample_observations_last, self.observations)
        next_observations["goal"] = hierarchy_selector(next_observations["goal"])
        terminals = np.ones([batch_size])

        # force the achieved goals to occur every goal_skip
        goal_idx = np.minimum(idx - self.terminals[idx, 0].astype(
            np.int32) % goal_skip + goal_skip, self.max_num_steps)
        next_goal_idx = np.minimum(next_idx - self.terminals[next_idx, 0].astype(
            np.int32) % goal_skip + goal_skip, self.max_num_steps)

        # sample observation goals achieved by the agent
        def sample_goals(data):
            return data[goal_idx, ...]

        # sample observation goals achieved by the agent
        def sample_goals_last(data):
            return data[next_goal_idx, ...]

        # sample current batch from a nested structure
        achieved_goals = nested_apply(sample_goals, self.observations)
        observations["achieved_goal"] = achieved_goals

        # sample current batch from a nested structure
        achieved_next_goals = nested_apply(sample_goals_last, self.observations)
        next_observations["achieved_goal"] = achieved_next_goals

        # return the samples in a batch
        return observations, actions, rewards, next_observations, terminals
