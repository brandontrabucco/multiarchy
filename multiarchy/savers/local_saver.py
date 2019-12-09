"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.savers.saver import Saver
import tensorflow as tf
import os
import pickle as pkl


class LocalSaver(Saver):

    def __init__(
        self,
        replay_buffer,
        logging_dir,
        **models
    ):
        # create a saver instance that saves models to the disk
        self.replay_buffer = replay_buffer
        self.logging_dir = logging_dir
        self.models = models

        tf.io.gfile.makedirs(logging_dir)

    def save(
        self,
    ):
        # save the replay buffer and the neural network states
        with open(os.path.join(
                self.logging_dir, "replay_buffer.pkl"), "wb") as f:
            pkl.dump(self.replay_buffer.to_dict(), f)

        # save many tensorflow keras models to the disk
        for name, model in self.models.items():
            model.save_weights(
                os.path.join(self.logging_dir, name + ".ckpt"))

    def load(
        self,
    ):
        # load the replay buffer and the neural network states
        if os.path.exists(os.path.join(
                self.logging_dir, "replay_buffer.pkl")):
            with open(os.path.join(
                    self.logging_dir, "replay_buffer.pkl"), "rb") as f:
                self.replay_buffer.from_dict(pkl.load(f))

        # load many tensorflow keras models from the disk
        for name, model in self.models.items():
            if os.path.exists(os.path.join(
                    self.logging_dir, name + ".ckpt")):
                model.load_weights(
                    os.path.join(self.logging_dir, name + ".ckpt"))
