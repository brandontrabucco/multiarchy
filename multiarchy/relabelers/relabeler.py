"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Relabeler(ABC):

    def __init__(
            self,
            replay_buffer
    ):
        # wrap a replay buffer and relabel samples from it
        self.replay_buffer = replay_buffer

    def __getattr__(
            self,
            attr
    ):
        # pass attribute lookups to the replay_buffer
        if hasattr(self, attr):
            return self.__dict__[attr]
        else:
            return getattr(self.replay_buffer, attr)

    def sample(
            self,
            batch_size,
            time_skip=1,
            goal_skip=1,
            hierarchy_selector=(lambda x: x)
    ):
        # relabel batches of data exiting the replay buffer
        return self.relabel(self.replay_buffer.sample(
            batch_size,
            time_skip=time_skip,
            goal_skip=goal_skip,
            hierarchy_selector=hierarchy_selector))

    @abstractmethod
    def relabel(
            self,
            batch_of_data
    ):
        # relabel batches of data exiting the replay buffer
        return NotImplemented
