"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.replay_buffers.replay_buffer import ReplayBuffer
from multiarchy.replay_buffers.remote.path_replay_buffer import RemotePathReplayBuffer
import ray


class PathReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            max_path_length=1000,
            max_num_paths=1000
    ):
        self.replay_buffer = RemotePathReplayBuffer.remote(
            max_path_length=max_path_length,
            max_num_paths=max_num_paths)

    def empty(
            self
    ):
        # empties the replay buffer of its elements
        return ray.get(self.replay_buffer.empty.remote())

    def get_total_paths(
            self
    ):
        # return the total number of episodes collected
        return ray.get(self.replay_buffer.get_total_paths.remote())

    def get_total_steps(
            self
    ):
        # return the total number of transitions collected
        return ray.get(self.replay_buffer.get_total_steps.remote())

    def to_dict(
            self,
    ):
        # save the replay buffer to a dictionary
        return ray.get(self.replay_buffer.to_dict.remote())

    def from_dict(
            self,
            state
    ):
        # load the replay buffer from a dictionary
        ray.get(self.replay_buffer.from_dict.remote(state))

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # insert a path into the replay buffer
        ray.get(self.replay_buffer.insert_path.remote(
            observations,
            actions,
            rewards))

    def sample(
            self,
            batch_size,
            time_skip=1,
            hierarchy_selector=(lambda x: x)
    ):
        # determine which steps to sample from
        return ray.get(self.replay_buffer.sample.remote(
            batch_size, time_skip=time_skip, hierarchy_selector=hierarchy_selector))
