"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from multiarchy.distributions.distribution import Distribution
import tensorflow as tf


class Categorical(Distribution):

    def __init__(
            self,
            model,
            temp=1.0,
            tau=0.01,
            optimizer_class=tf.keras.optimizers.Adam,
            optimizer_kwargs=None,
    ):
        # create a categorical distribution with fixed or learned temperature
        Distribution.__init__(
            self, model,
            tau=tau,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs)
        self.temp = temp

    def clone(
            self
    ):
        # create an exact duplicate (different pointers) of the policy
        return Categorical(
            tf.keras.models.clone_model(self.model),
            temp=self.temp,
            tau=self.tau,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs)

    def get_parameters(
            self,
            *inputs
    ):
        # get the log probabilities of the categorical distribution
        x = self.model(tf.concat(inputs, (-1)))
        if self.temp is None:
            logits, temp = x[..., :(-1)], x[..., (-1):]
        else:
            logits, temp = x, self.temp
        return tf.math.log_softmax(x / temp)

    def sample(
            self,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        logits = self.get_parameters(*inputs)

        # sample from the categorical distribution
        categorical_samples = tf.reshape(
            tf.random.categorical(
                tf.reshape(logits, [-1, tf.shape(logits)[(-1)]]), 1),
            tf.shape(logits)[:(-1)])

        # compute the log probability density of the samples
        return categorical_samples, tf.gather_nd(
            logits, categorical_samples, batch_dims=len(categorical_samples.shape))

    def expected_value(
            self,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        logits = self.get_parameters(*inputs)

        # sample from the categorical distribution
        categorical_samples = tf.argmax(logits, axis=(-1), output_type=tf.int32)

        # compute the log probability density of the mean
        return categorical_samples, tf.gather_nd(
            logits, categorical_samples, batch_dims=len(categorical_samples.shape))

    def log_prob(
            self,
            categorical_samples,
            *inputs
    ):
        # get the mean and the log standard deviation of the distribution
        logits = self.get_parameters(*inputs)

        # compute the log probability density of the samples
        return tf.gather_nd(
            logits, categorical_samples, batch_dims=len(categorical_samples.shape))