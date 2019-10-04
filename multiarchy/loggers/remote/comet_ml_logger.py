"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.loggers.logger import Logger
import ray


@ray.remote
class RemoteCometMLLogger(Logger):

    def __init__(
        self,
        replay_buffer,
        http_address,
    ):
        # create a separate tensor board logging thread
        self.replay_buffer = replay_buffer
        self.http_address = http_address

    def record(
        self,
        key,
        value,
    ):
        # get the current number of samples collected
        step = ray.get(self.replay_buffer.get_total_steps.remote())

        # generate a plot and write the plot to tensor board
        if len(value.shape) == 1:
            pass

        # generate several plots and write the plot to tensor board
        elif len(value.shape) == 2:
            pass

        # write a single image to tensor board
        elif len(value.shape) == 3:
            pass

        # write several images to tensor board
        elif len(value.shape) == 4:
            pass

        # otherwise, assume the tensor is still a scalar
        else:
            pass
