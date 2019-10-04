"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.loggers.logger import Logger
from multiarchy.loggers.remote.tensorboard_logger import RemoteTensorboardLogger


class TensorboardLogger(Logger):

    def __init__(
        self,
        replay_buffer,
        logging_dir,
    ):
        # create a separate tensor board logging thread
        self.logger = RemoteTensorboardLogger.remote(replay_buffer, logging_dir)

    def record(
        self,
        key,
        value,
    ):
        # get the current number of samples collected
        self.logger.record.remote(key, value)
