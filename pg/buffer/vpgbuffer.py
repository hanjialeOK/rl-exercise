import numpy as np


EPS = 1e-8


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, ) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, rew, done, val, logp):
        """
        Store transition.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0.):
        assert self.ptr == self.max_size
        rews = np.append(self.rew_buf, last_val)
        vals = np.append(self.val_buf, last_val)

        # the next two" lines implement GAE-Lambda advantage calculation
        lastgaelam = 0.0
        lastret = rews[-1]
        for t in reversed(range(self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            delta = rews[t] + self.gamma * nondone * vals[t + 1] - vals[t]
            self.adv_buf[t] = lastgaelam = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            self.ret_buf[t] = lastret = rews[t] + \
                self.gamma * nondone * lastret

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / \
            (np.std(self.adv_buf) + EPS)
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf]


class TRPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, ) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mu_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.logstd_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, rew, done, val, logp, mu, logstd):
        """
        Store transition.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mu_buf[self.ptr] = mu
        self.logstd_buf[self.ptr] = logstd
        self.ptr += 1

    def finish_path(self, last_val=0.):
        assert self.ptr == self.max_size
        rews = np.append(self.rew_buf, last_val)
        vals = np.append(self.val_buf, last_val)

        # the next two" lines implement GAE-Lambda advantage calculation
        lastgaelam = 0.0
        lastret = rews[-1]
        for t in reversed(range(self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            max_kl = rews[t] + self.gamma * nondone * vals[t + 1] - vals[t]
            self.adv_buf[t] = lastgaelam = max_kl + \
                self.gamma * self.lam * nondone * lastgaelam
            self.ret_buf[t] = lastret = rews[t] + \
                self.gamma * nondone * lastret

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / \
            (np.std(self.adv_buf) + EPS)
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf, self.mu_buf, self.logstd_buf]
