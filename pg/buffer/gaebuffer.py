import numpy as np


EPS = 1e-8


class GAEBuffer:
    def __init__(self, obs_dim, act_dim, size,  gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, ) + obs_dim, dtype=np.float64)
        self.act_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((size, ), dtype=np.float32)
        self.rew_buf = np.zeros((size, ), dtype=np.float32)
        self.done_buf = np.zeros((size, ), dtype=np.float32)
        self.ret_buf = np.zeros((size, ), dtype=np.float32)
        self.val_buf = np.zeros((size+1, ), dtype=np.float32)
        self.logp_buf = np.zeros((size, ), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.path_start_idx = 0

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        assert obs.shape == self.obs_dim
        assert act.shape == self.act_dim
        # assert rew.shape == (1,)
        # assert done.shape == (1,)
        # assert val.shape == (1,)
        # assert logp.shape == (1,)
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=None):
        # assert self.ptr == self.max_size
        # assert last_val.shape == (self.num_env,)
        path_slice = slice(self.path_start_idx, self.ptr)
        # vals = np.append(self.val_buf[path_slice], last_val)
        self.val_buf[self.ptr] = last_val

        # GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(self.path_start_idx, self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            delta = self.rew_buf[t] + self.gamma * \
                nondone * self.val_buf[t + 1] - self.val_buf[t]
            self.adv_buf[t] = lastgaelam = delta + \
                self.gamma * self.lam * nondone * lastgaelam
        self.ret_buf[path_slice] = self.adv_buf[path_slice] + \
            self.val_buf[path_slice]

        self.path_start_idx = self.ptr
        pass

    def get(self):
        assert self.ptr == self.max_size
        return [self.obs_buf.reshape(self.max_size, -1),
                self.act_buf.reshape(self.max_size, -1),
                self.adv_buf.reshape(-1),
                self.ret_buf.reshape(-1),
                self.logp_buf.reshape(-1),
                self.val_buf.reshape(-1)[:-1]]

    def get_rms_data(self):
        assert self.ptr == self.max_size
        return [self.obs_buf.reshape(self.max_size, -1),
                self.ret_buf.reshape(-1)]

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0


class PPODistVBuffer:
    def __init__(self, obs_dim, act_dim, size, num_env=1,  gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, num_env) + obs_dim, dtype=np.float64)
        self.act_buf = np.zeros((size, num_env) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((size, num_env), dtype=np.float32)
        self.rew_buf = np.zeros((size, num_env), dtype=np.float32)
        self.done_buf = np.zeros((size, num_env), dtype=np.float32)
        self.ret_buf = np.zeros((size, num_env) + (32, ), dtype=np.float32)
        self.val_buf = np.zeros((size, num_env), dtype=np.float32)
        self.logp_buf = np.zeros((size, num_env), dtype=np.float32)
        self.distv_buf = np.zeros((size, num_env) + (32, ), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_env = num_env

    def store(self, obs, act, rew, done, val, logp, distv):
        """
        Store transition.
        """
        assert self.ptr < self.max_size
        assert obs.shape == (self.num_env,) + self.obs_dim
        assert act.shape == (self.num_env,) + self.act_dim
        assert rew.shape == (self.num_env,)
        assert done.shape == (self.num_env,)
        assert val.shape == (self.num_env,)
        assert logp.shape == (self.num_env,)
        assert distv.shape == (self.num_env,) + (32, )
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.distv_buf[self.ptr] = distv
        self.ptr += 1

    def finish_path(self, last_val=None):
        assert self.ptr == self.max_size
        assert last_val.shape == (self.num_env,)
        vals = np.append(self.val_buf, last_val.reshape(1, -1), axis=0)

        # the next two" lines implement GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            delta = self.rew_buf[t] + self.gamma * \
                nondone * vals[t + 1] - vals[t]
            self.adv_buf[t] = lastgaelam = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            # self.ret_buf[t] = lastret = rews[t] + \
            #     self.gamma * nondone * lastret
        self.ret_buf = self.adv_buf[:, :, None] + self.distv_buf
        pass

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        return [self.obs_buf.reshape(self.max_size * self.num_env, -1),
                self.act_buf.reshape(self.max_size * self.num_env, -1),
                self.adv_buf.reshape(-1),
                self.ret_buf.reshape(self.max_size, 32),
                self.logp_buf.reshape(-1),
                self.val_buf.reshape(-1)]


class PPOTD0Buffer:
    def __init__(self, obs_dim, act_dim, size, num_env=1,  gamma=0.99, lam=0.95):
        self.obs1_buf = np.zeros((size, num_env) + obs_dim, dtype=np.float64)
        self.obs2_buf = np.zeros((size, num_env) + obs_dim, dtype=np.float64)
        self.act_buf = np.zeros((size, num_env) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((size, num_env), dtype=np.float32)
        self.rew_buf = np.zeros((size, num_env), dtype=np.float32)
        self.done_buf = np.zeros((size, num_env), dtype=np.float32)
        self.ret_buf = np.zeros((size, num_env), dtype=np.float32)
        self.val_buf = np.zeros((size, num_env), dtype=np.float32)
        self.logp_buf = np.zeros((size, num_env), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_env = num_env

    def store(self, obs1, obs2, act, rew, done, val, logp):
        """
        Store transition.
        """
        assert self.ptr < self.max_size
        assert obs1.shape == (self.num_env,) + self.obs_dim
        assert obs2.shape == (self.num_env,) + self.obs_dim
        assert act.shape == (self.num_env,) + self.act_dim
        assert rew.shape == (self.num_env,)
        assert done.shape == (self.num_env,)
        assert val.shape == (self.num_env,)
        assert logp.shape == (self.num_env,)
        self.obs1_buf[self.ptr] = obs1
        self.obs2_buf[self.ptr] = obs2
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def finish_path(self, last_val=None):
        assert self.ptr == self.max_size
        assert last_val.shape == (self.num_env,)
        vals = np.append(self.val_buf, last_val.reshape(1, -1), axis=0)

        # the next two" lines implement GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            delta = self.rew_buf[t] + self.gamma * \
                nondone * vals[t + 1] - vals[t]
            self.adv_buf[t] = lastgaelam = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            # self.ret_buf[t] = lastret = rews[t] + \
            #     self.gamma * nondone * lastret
        self.ret_buf = self.adv_buf + self.val_buf
        pass

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        return [self.obs_buf.reshape(self.max_size * self.num_env, -1),
                self.act_buf.reshape(self.max_size * self.num_env, -1),
                self.adv_buf.reshape(-1),
                self.ret_buf.reshape(-1),
                self.logp_buf.reshape(-1),
                self.val_buf.reshape(-1)]


class TRPOBuffer(GAEBuffer):
    def __init__(self, obs_dim, act_dim, size, num_env=1, gamma=0.99, lam=0.95):
        self.mu_buf = np.zeros((size, num_env) + act_dim, dtype=np.float32)
        self.logstd_buf = np.zeros((1,) + act_dim, dtype=np.float32)
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, size=size,
                         num_env=num_env, gamma=gamma, lam=lam)

    def store(self, obs, act, rew, done, val, logp, mu, logstd):
        assert mu.shape == (self.num_env,) + self.act_dim
        assert logstd.shape == (1,) + self.act_dim
        self.mu_buf[self.ptr] = mu
        self.logstd_buf = logstd
        super().store(obs, act, rew, done, val, logp)

    def finish_path(self, last_val=None):
        super().finish_path(last_val=last_val)

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        return [self.obs_buf.reshape(self.max_size * self.num_env, -1),
                self.act_buf.reshape(self.max_size * self.num_env, -1),
                self.adv_buf.reshape(-1),
                self.ret_buf.reshape(-1),
                self.logp_buf.reshape(-1),
                self.val_buf.reshape(-1),
                self.mu_buf.reshape(self.max_size * self.num_env, -1),
                self.logstd_buf]
