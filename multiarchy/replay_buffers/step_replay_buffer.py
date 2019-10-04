"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.replay_buffers.replay_buffer import ReplayBuffer
from multiarchy.replay_buffers.remote.step_replay_buffer import RemoteStepReplayBuffer
import ray


class StepReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            max_num_steps=1000000,
            selector=None
    ):
        self.replay_buffer = RemoteStepReplayBuffer.remote(
            max_num_steps=max_num_steps,
            selector=selector)

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
        self.replay_buffer.from_dict.remote(state)

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # insert a path into the replay buffer
        self.replay_buffer.insert_path.remote(
            observations,
            actions,
            rewards)

    def sample(
            self,
            batch_size
    ):
        # determine which steps to sample from
        return ray.get(self.replay_buffer.sample.remote(batch_size))
