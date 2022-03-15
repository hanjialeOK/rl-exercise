import tensorflow as tf
import numpy as np


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class TfRunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    '''
    TensorFlow variables-based implmentation of computing running mean and std
    Benefit of this implementation is that it can be saved / loaded together with the tensorflow model
    '''

    def __init__(self, epsilon=1e-4, shape=(), scope=''):
        sess = tf.compat.v1.get_default_session()

        self._new_mean = tf.compat.v1.placeholder(
            shape=shape, dtype=tf.float64)
        self._new_var = tf.compat.v1.placeholder(shape=shape, dtype=tf.float64)
        self._new_count = tf.compat.v1.placeholder(shape=(), dtype=tf.float64)

        with tf.compat.v1.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self._mean = tf.compat.v1.get_variable('mean',  initializer=np.zeros(
                shape, 'float64'),      dtype=tf.float64)
            self._var = tf.compat.v1.get_variable('std',   initializer=np.ones(
                shape, 'float64'),       dtype=tf.float64)
            self._count = tf.compat.v1.get_variable('count', initializer=np.full(
                (), epsilon, 'float64'), dtype=tf.float64)

        self.update_ops = tf.group([
            self._var.assign(self._new_var),
            self._mean.assign(self._new_mean),
            self._count.assign(self._new_count)
        ])

        sess.run(tf.compat.v1.variables_initializer(
            [self._mean, self._var, self._count]))
        self.sess = sess
        self._set_mean_var_count()

    def _set_mean_var_count(self):
        self.mean, self.var, self.count = self.sess.run(
            [self._mean, self._var, self._count])

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_mean, new_var, new_count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

        self.sess.run(self.update_ops, feed_dict={
            self._new_mean: new_mean,
            self._new_var: new_var,
            self._new_count: new_count
        })

        self._set_mean_var_count()
