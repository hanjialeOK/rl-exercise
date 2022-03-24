import numpy as np


class RunningMeanStd2(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, newValue):
        m2 = self.count * self.var
        self.count += 1
        delta = newValue - self.mean
        self.mean += delta / self.count
        delta2 = newValue - self.mean
        m2 += delta * delta2
        self.var = m2 / self.count


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


if __name__ == '__main__':
    shape = (17, )
    rms = RunningMeanStd(shape=shape)
    my_rms = RunningMeanStd2(shape=shape)
    data = np.random.random(size=(100, ) + shape)
    for i in range(data.shape[0]):
        rms.update(data[i].reshape(1, -1))
        my_rms.update(data[i])
    mean, var = rms.mean, rms.var
    my_mean, my_var = my_rms.mean, my_rms.var
    raise
