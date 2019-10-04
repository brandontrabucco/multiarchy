"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.algorithms.algorithm import Algorithm
import tensorflow as tf


class SAC(Algorithm):

    def __init__(
            self,
            policy,
            target_policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            replay_buffer,
            reward_scale=1.0,
            discount=0.99,
            target_clipping=0.5,
            target_noise=0.2,
            input_selector=(lambda x: x["observation"]),
            batch_size=32,
            update_every=1,
            update_after=1,
            logger=None,
            logging_prefix="sac/"
    ):
        # train a policy using the vanilla policy gradient
        Algorithm.__init__(
            self,
            replay_buffer,
            batch_size=batch_size,
            update_every=update_every,
            update_after=update_after,
            logger=logger,
            logging_prefix=logging_prefix)

        # each neural network is probabilistic
        self.policy = policy
        self.target_policy = target_policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        # select into the observation dictionary
        self.input_selector = input_selector

        # control some parameters that are important for sac
        self.reward_scale = reward_scale
        self.discount = discount
        self.target_clipping = target_clipping
        self.target_noise = target_noise

    def update_algorithm(
            self,
            observations,
            actions,
            rewards,
            next_observations,
            terminals
    ):
        # select from the observation dictionary
        observations = self.input_selector(observations)
        next_observations = self.input_selector(next_observations)

        # build a tape to collect gradients from the policy and critics
        with tf.GradientTape(persistent=True) as tape:
            mean_actions, log_pi = self.policy.expected_value(observations)
            next_mean_actions, next_log_pi = self.target_policy.expected_value(
                next_observations)

            # build the target policy noise
            noise = tf.clip_by_value(
                self.target_noise * tf.random.normal(tf.shape(mean_actions)),
                -self.target_clipping, self.target_clipping)
            next_noisy_actions = next_mean_actions + noise

            # build the q function target value
            inputs = tf.concat([next_observations, next_noisy_actions], -1)
            target_qf1_value = self.target_qf1(inputs)[..., 0]
            self.record("target_qf1_value", tf.reduce_mean(target_qf1_value))
            target_qf2_value = self.target_qf2(inputs)[..., 0]
            self.record("target_qf2_value", tf.reduce_mean(target_qf2_value))
            qf_targets = tf.stop_gradient(
                self.reward_scale * rewards + terminals * self.discount * (
                    tf.minimum(target_qf1_value, target_qf2_value)))
            self.record("qf_targets", tf.reduce_mean(qf_targets))

            # build the q function loss
            inputs = tf.concat([observations, actions], -1)
            qf1_value = self.qf1(inputs)[..., 0]
            self.record("qf1_value", tf.reduce_mean(qf1_value))
            qf2_value = self.qf2(inputs)[..., 0]
            self.record("qf2_value", tf.reduce_mean(qf2_value))
            qf1_loss = tf.reduce_mean(tf.keras.losses.logcosh(qf_targets, qf1_value))
            self.record("qf1_loss", qf1_loss)
            qf2_loss = tf.reduce_mean(tf.keras.losses.logcosh(qf_targets, qf2_value))
            self.record("qf2_loss", qf2_loss)

            # build the policy loss
            inputs = tf.concat([observations, mean_actions], -1)
            policy_qf1_value = self.qf1(inputs)[..., 0]
            self.record("policy_qf1_value", tf.reduce_mean(policy_qf1_value))
            policy_qf2_value = self.qf2(inputs)[..., 0]
            self.record("policy_qf2_value", tf.reduce_mean(policy_qf2_value))
            policy_loss = -tf.reduce_mean(
                tf.minimum(policy_qf1_value, policy_qf2_value))
            self.record("policy_loss", policy_loss)

        # back prop gradients
        self.qf1.apply_gradients(
            self.qf1.compute_gradients(qf1_loss, tape))
        self.qf2.apply_gradients(
            self.qf2.compute_gradients(qf2_loss, tape))
        self.policy.apply_gradients(
            self.policy.compute_gradients(policy_loss, tape))

        # soft update target parameters
        self.target_policy.soft_update(self.policy.get_weights())
        self.target_qf1.soft_update(self.qf1.get_weights())
        self.target_qf2.soft_update(self.qf2.get_weights())
