import numpy as np
import tensorflow as tf

class DiagGaussianPd():
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def neglogp(self, x):
        return 0.5 * tf.square((x - self.mean) / self.std) + 0.5 * np.log(2.0 * np.pi) + self.logstd
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5
    def entropy(self):
        return self.logstd + .5 * np.log(2.0 * np.pi * np.e)
    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
