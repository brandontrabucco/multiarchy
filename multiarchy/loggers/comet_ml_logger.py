"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.loggers.logger import Logger
from multiarchy.loggers.remote.comet_ml_logger import RemoteCometMLLogger


class CometMLLogger(Logger):

    def __init__(
        self,
        replay_buffer,
        http_address,
    ):
        # create a separate tensor board logging thread
        self.logger = RemoteCometMLLogger.remote(replay_buffer, http_address)

    def record(
        self,
        key,
        value,
    ):
        # get the current number of samples collected
        self.logger.record.remote(key, value)
