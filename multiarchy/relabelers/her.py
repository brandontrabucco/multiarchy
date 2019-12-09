"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.relabelers.relabeler import Relabeler
import tensorflow as tf


class HER(Relabeler):

    def __init__(
            self,
            replay_buffer,
            relabel_probability=1.0,
            observation_key="observation"
    ):
        # wrap a replay buffer and relabel samples from it
        Relabeler.__init__(self, replay_buffer)
        self.relabel_probability = relabel_probability
        self.observation_key = observation_key

    def relabel(
            self,
            batch_of_data
    ):
        if len(batch_of_data) == 4:
            # the batch is using a path replay buffer notation
            observation, actions, rewards, terminals = batch_of_data
            mask = tf.math.less_equal(
                tf.random.uniform(
                    tf.shape(rewards)[:2],
                    maxval=1.0,
                    dtype=tf.float32), self.relabel_probability)

            original_goals = observation["goals"]
            while len(mask.shape) < len(original_goals.shape):
                mask = tf.expand_dims(mask, -1)

            achieved_goals = observation["achieved_goals"][self.observation_key]
            observation["goals"] = tf.where(
                mask, achieved_goals, original_goals)
            return observation, actions, rewards, terminals

        elif len(batch_of_data) == 5:
            # the batch is using a step replay buffer notation
            observation, actions, rewards, terminals, next_observation = batch_of_data
            mask = tf.math.less_equal(
                tf.random.uniform(
                    tf.shape(rewards)[:1],
                    maxval=1.0,
                    dtype=tf.float32), self.relabel_probability)

            original_goals = observation["goals"]
            next_original_goals = next_observation["goals"]
            while len(mask.shape) < len(original_goals.shape):
                mask = tf.expand_dims(mask, -1)

            achieved_goals = observation["achieved_goals"][self.observation_key]
            observation["goals"] = tf.where(
                mask, achieved_goals, original_goals)

            next_achieved_goals = next_observation["achieved_goals"][self.observation_key]
            next_observation["goals"] = tf.where(
                mask, next_achieved_goals, next_original_goals)
            return observation, actions, rewards, terminals, next_observation
