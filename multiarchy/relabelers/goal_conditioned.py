"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.relabelers.relabeler import Relabeler
import tensorflow as tf


class GoalConditioned(Relabeler):

    def __init__(
            self,
            replay_buffer,
            order=2,
            goal_conditioned_scale=1.0,
            reward_scale=0.0,
            observation_key="observation"
    ):
        # wrap a replay buffer and relabel samples from it
        Relabeler.__init__(self, replay_buffer)
        self.order = order
        self.goal_conditioned_scale = goal_conditioned_scale
        self.reward_scale = reward_scale
        self.observation_key = observation_key

    def relabel(
            self,
            batch_of_data
    ):
        if len(batch_of_data) == 4:
            # the batch is using a path replay buffer notation
            observation, actions, rewards, terminals = batch_of_data
            batch_dim = tf.shape(rewards)[0]
            max_path_length = tf.shape(rewards)[1]

            goals = tf.reshape(observation["goals"], [batch_dim, max_path_length, -1])
            states = tf.reshape(
                observation[self.observation_key], [batch_dim, max_path_length, -1])

            goal_conditioned_reward = -tf.pad(tf.linalg.norm(
                states - goals, ord=self.order, axis=(-1)), [[0, 0], [0, 1]])[0, 1:]
            mixture = (self.reward_scale * rewards +
                       self.goal_conditioned_scale * goal_conditioned_reward)

            return observation, actions, mixture, terminals

        elif len(batch_of_data) == 5:
            # the batch is using a step replay buffer notation
            observation, actions, rewards, terminals, next_observation = batch_of_data
            batch_dim = tf.shape(rewards)[0]

            goals = tf.reshape(observation["goals"], [batch_dim, -1])
            states = tf.reshape(next_observation[self.observation_key], [batch_dim, -1])

            goal_conditioned_reward = -tf.linalg.norm(
                states - goals, ord=self.order, axis=(-1))
            mixture = (self.reward_scale * rewards +
                       self.goal_conditioned_scale * goal_conditioned_reward)

            return observation, actions, mixture, terminals, next_observation
