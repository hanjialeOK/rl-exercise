import numpy as np


class GAEBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, env, horizon, gamma=0.99, lam=0.95, compute_v=None):
        obs_shape = env.observation_space.shape
        ac_shape = env.action_space.shape
        obs_dtype = env.observation_space.dtype.name
        ac_dtype = env.action_space.dtype.name
        self.obs_buf = np.zeros((horizon, ) + obs_shape, dtype=obs_dtype)
        self.obs2_buf = np.zeros((horizon, ) + obs_shape, dtype=obs_dtype)
        self.ac_buf = np.zeros((horizon, ) + ac_shape, dtype=ac_dtype)
        self.rew_buf = np.zeros((horizon, ), dtype=np.float32)
        self.done_buf = np.zeros((horizon, ), dtype=np.float32)
        self.ret_buf = np.zeros((horizon, ), dtype=np.float32)
        self.adv_buf = np.zeros((horizon, ), dtype=np.float32)
        self.val_buf = np.zeros((horizon, ), dtype=np.float32)
        self.next_val_buf = np.zeros((horizon, ), dtype=np.float32)
        self.logp_buf = np.zeros((horizon, ), dtype=np.float32)
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.max_size = horizon
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = self.ptr
        self.compute_v = compute_v

    def store(self, obs, ac, rew, done, obs2, val, logp, *args):
        assert self.ptr < self.max_size
        # assert obs.shape == self.obs_shape
        # assert ac.shape == self.ac_shape
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.ac_buf[self.ptr] = ac
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self):
        start = self.path_start_idx

        self.next_val_buf[start:self.ptr-1] = self.val_buf[start+1:self.ptr]

        last_done = self.done_buf[self.ptr-1]
        last_ob2 = self.obs2_buf[self.ptr-1]
        last_val = 0. if last_done else self.compute_v(last_ob2)
        self.next_val_buf[self.ptr-1] = last_val

        # GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(self.path_start_idx, self.ptr)):
            delta = self.rew_buf[t] + \
                self.gamma * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = \
                delta + self.gamma * self.lam * lastgaelam
            lastgaelam = self.adv_buf[t]

        self.ret_buf[start:self.ptr] = self.adv_buf[start:self.ptr] + \
            self.val_buf[start:self.ptr]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size

        # Reset ptr
        self.ptr = 0
        self.path_start_idx = self.ptr

        return [self.obs_buf,
                self.ac_buf,
                self.adv_buf,
                self.ret_buf,
                self.val_buf,
                self.logp_buf]

    def get_rms_data(self):
        assert self.ptr == self.max_size
        return [self.obs_buf,
                self.ret_buf]

    def reset(self):
        self.ptr = 0
        self.path_start_idx = self.ptr


class GAEVBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, env, horizon, nlatest=1, gamma=0.99, lam=0.95,
                 compute_v_pik=None, compute_logp_pik=None):
        obs_shape = env.observation_space.shape
        ac_shape = env.action_space.shape
        obs_dtype = env.observation_space.dtype.name
        ac_dtype = env.action_space.dtype.name
        max_size = horizon * nlatest
        self.obs_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.obs2_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.raw_obs_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.raw_obs2_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.ac_buf = np.zeros((max_size, ) + ac_shape, dtype=ac_dtype)
        self.rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.raw_rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.done_buf = np.zeros((max_size, ), dtype=np.float32)
        self.trun_buf = np.zeros((max_size, ), dtype=np.float32)
        self.ret_buf = np.zeros((max_size, ), dtype=np.float32)
        self.adv_buf = np.zeros((max_size, ), dtype=np.float32)
        self.val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.next_val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.logp_buf = np.zeros((max_size, ), dtype=np.float32)
        self.logp_pik_buf = np.zeros((max_size, ), dtype=np.float32)
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.max_size = max_size
        self.horizon = horizon
        self.nlatest = nlatest
        self.gamma = gamma
        self.lam = lam
        self.ptr = self.max_size - self.horizon
        self.path_start_idx = self.ptr
        self.count = 0
        self.obfilt = env._obfilt
        self.rewfilt = env._rewfilt
        self.compute_v_pik = compute_v_pik
        self.compute_logp_pik = compute_logp_pik

    def store(self, obs, ac, rew, done, obs2, val, logp, raw_obs, raw_rew, raw_obs2):
        assert self.ptr < self.max_size
        # assert obs.shape == self.obs_shape
        # assert raw_obs.shape == self.obs_shape
        # assert ac.shape == self.ac_shape
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.ac_buf[self.ptr] = ac
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.trun_buf[self.ptr] = done
        self.raw_obs_buf[self.ptr] = raw_obs
        self.raw_obs2_buf[self.ptr] = raw_obs2
        self.raw_rew_buf[self.ptr] = raw_rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        self.count = min(self.count+1, self.max_size)

    def finish_path(self):
        self.trun_buf[self.ptr-1] = 1

        self.path_start_idx = self.ptr

    def vtrace(self):
        assert self.ptr == self.max_size
        assert self.count % self.horizon == 0 and self.count > 0

        if self.obfilt:
            self.obs_buf[:] = self.obfilt(self.raw_obs_buf)
            self.obs2_buf[:] = self.obfilt(self.raw_obs2_buf)
        if self.rewfilt:
            self.rew_buf[:] = self.rewfilt(self.raw_rew_buf)

        self.val_buf[:] = self.compute_v_pik(self.obs_buf)
        self.next_val_buf[:] = self.compute_v_pik(self.obs2_buf)
        self.logp_pik_buf[:] = self.compute_logp_pik(self.obs_buf, self.ac_buf)

        rho = np.exp(self.logp_pik_buf - self.logp_buf)
        # Reduce bias here!
        rho = np.minimum(rho, 1.0)

        lastgaelam = 0.0
        for t in reversed(range(self.max_size - self.count, self.max_size)):
            nondone = 1.0 - self.done_buf[t]
            nontruncated = 1.0 - self.trun_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * nondone * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            lastgaelam = rho[t] * self.adv_buf[t]
        self.ret_buf[:] = self.adv_buf * rho + self.val_buf

        # Reset ptr
        self.ptr = self.max_size - self.horizon
        self.path_start_idx = self.ptr

        return [self.obs_buf[-self.count:],
                self.ac_buf[-self.count:],
                self.adv_buf[-self.count:],
                self.ret_buf[-self.count:],
                self.val_buf[-self.count:],
                self.logp_buf[-self.count:],
                self.logp_pik_buf[-self.count:]]

    def update(self):
        tail = self.max_size - self.horizon
        head = self.horizon
        if not self.obfilt:
            self.obs_buf[:tail] = self.obs_buf[head:]
            self.obs2_buf[:tail] = self.obs2_buf[head:]
        if not self.rewfilt:
            self.rew_buf[:tail] = self.rew_buf[head:]
        self.ac_buf[:tail] = self.ac_buf[head:]
        self.done_buf[:tail] = self.done_buf[head:]
        self.trun_buf[:tail] = self.trun_buf[head:]
        # self.ret_buf[:tail] = self.ret_buf[head:]
        # self.adv_buf[:tail] = self.adv_buf[head:]
        # self.val_buf[:tail] = self.val_buf[head:]
        # self.next_val_buf[:tail] = self.next_val_buf[head:]
        self.logp_buf[:tail] = self.logp_buf[head:]
        # self.logp_pik_buf[:tail] = self.logp_pik_buf[head:]
        if self.obfilt:
            self.raw_obs_buf[:tail] = self.raw_obs_buf[head:]
            self.raw_obs2_buf[:tail] = self.raw_obs2_buf[head:]
        if self.rewfilt:
            self.raw_rew_buf[:tail] = self.raw_rew_buf[head:]

    def get_rms_data(self):
        assert self.ptr == self.max_size
        # Return the latest RMS data
        return [self.obs_buf[-self.horizon:],
                self.ret_buf[-self.horizon:]]

    def reset(self):
        self.ptr = self.max_size - self.horizon
        self.path_start_idx = self.ptr
        self.count = 0


class DISCBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, env, horizon, nlatest=1, gamma=0.99, lam=0.95,
                 compute_v_pik=None, compute_logp_pik=None):
        obs_shape = env.observation_space.shape
        ac_shape = env.action_space.shape
        obs_dtype = env.observation_space.dtype.name
        ac_dtype = env.action_space.dtype.name
        max_size = horizon * nlatest
        self.obs_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.obs2_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.raw_obs_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.raw_obs2_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.ac_buf = np.zeros((max_size, ) + ac_shape, dtype=ac_dtype)
        self.rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.raw_rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.done_buf = np.zeros((max_size, ), dtype=np.float32)
        self.trun_buf = np.zeros((max_size, ), dtype=np.float32)
        self.ret_buf = np.zeros((max_size, ), dtype=np.float32)
        self.adv_buf = np.zeros((max_size, ), dtype=np.float32)
        self.val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.next_val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.logp_disc_buf = np.zeros((max_size, ) + ac_shape, dtype=np.float32)
        self.logp_disc_pik_buf = np.zeros((max_size, ) + ac_shape, dtype=np.float32)
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.max_size = max_size
        self.horizon = horizon
        self.nlatest = nlatest
        self.gamma = gamma
        self.lam = lam
        self.ptr = self.max_size - self.horizon
        self.path_start_idx = self.ptr
        self.count = 0
        self.obfilt = env._obfilt
        self.rewfilt = env._rewfilt
        self.compute_v_pik = compute_v_pik
        self.compute_logp_pik = compute_logp_pik

    def store(self, obs, ac, rew, done, obs2, val, logp, raw_obs, raw_rew, raw_obs2):
        assert self.ptr < self.max_size
        # assert obs.shape == self.obs_shape
        # assert raw_obs.shape == self.obs_shape
        # assert ac.shape == self.ac_shape
        # assert logp.shape == self.ac_shape
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.ac_buf[self.ptr] = ac
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.trun_buf[self.ptr] = done
        self.raw_obs_buf[self.ptr] = raw_obs
        self.raw_obs2_buf[self.ptr] = raw_obs2
        self.raw_rew_buf[self.ptr] = raw_rew
        self.val_buf[self.ptr] = val
        self.logp_disc_buf[self.ptr] = logp
        self.ptr += 1
        self.count = min(self.count+1, self.max_size)

    def finish_path(self):
        self.trun_buf[self.ptr-1] = 1

        self.path_start_idx = self.ptr

    def vtrace(self):
        assert self.ptr == self.max_size
        assert self.count % self.horizon == 0 and self.count > 0

        if self.obfilt:
            self.obs_buf[:] = self.obfilt(self.raw_obs_buf)
            self.obs2_buf[:] = self.obfilt(self.raw_obs2_buf)
        if self.rewfilt:
            self.rew_buf[:] = self.rewfilt(self.raw_rew_buf)

        self.val_buf[:] = self.compute_v_pik(self.obs_buf)
        self.next_val_buf[:] = self.compute_v_pik(self.obs2_buf)
        self.logp_disc_pik_buf[:] = self.compute_logp_pik(self.obs_buf, self.ac_buf)

        logp_pik = np.sum(self.logp_disc_pik_buf, axis=1)
        logp_old = np.sum(self.logp_disc_buf, axis=1)

        rho_ = np.exp(logp_pik - logp_old)
        # Reduce bias here!
        rho = np.minimum(rho_, 1.0)

        lastgaelam = 0.0
        for t in reversed(range(self.max_size - self.count, self.max_size)):
            nondone = 1.0 - self.done_buf[t]
            nontruncated = 1.0 - self.trun_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * nondone * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            lastgaelam = rho[t] * self.adv_buf[t]
        self.ret_buf[:] = self.adv_buf * rho + self.val_buf

        # Reset ptr
        self.ptr = self.max_size - self.horizon
        self.path_start_idx = self.ptr

        return [self.obs_buf[-self.count:],
                self.ac_buf[-self.count:],
                self.adv_buf[-self.count:],
                self.ret_buf[-self.count:],
                self.val_buf[-self.count:],
                self.logp_disc_buf[-self.count:],
                self.logp_disc_pik_buf[-self.count:]]

    def update(self):
        tail = self.max_size - self.horizon
        head = self.horizon
        if not self.obfilt:
            self.obs_buf[:tail] = self.obs_buf[head:]
            self.obs2_buf[:tail] = self.obs2_buf[head:]
        if not self.rewfilt:
            self.rew_buf[:tail] = self.rew_buf[head:]
        self.ac_buf[:tail] = self.ac_buf[head:]
        self.done_buf[:tail] = self.done_buf[head:]
        self.trun_buf[:tail] = self.trun_buf[head:]
        # self.ret_buf[:tail] = self.ret_buf[head:]
        # self.adv_buf[:tail] = self.adv_buf[head:]
        # self.val_buf[:tail] = self.val_buf[head:]
        # self.next_val_buf[:tail] = self.next_val_buf[head:]
        self.logp_disc_buf[:tail] = self.logp_disc_buf[head:]
        # self.logp_pik_buf[:tail] = self.logp_pik_buf[head:]
        if self.obfilt:
            self.raw_obs_buf[:tail] = self.raw_obs_buf[head:]
            self.raw_obs2_buf[:tail] = self.raw_obs2_buf[head:]
        if self.rewfilt:
            self.raw_rew_buf[:tail] = self.raw_rew_buf[head:]

    def get_rms_data(self):
        assert self.ptr == self.max_size
        # Return the latest RMS data
        return [self.obs_buf[-self.horizon:],
                self.ret_buf[-self.horizon:]]

    def reset(self):
        self.ptr = self.max_size - self.horizon
        self.path_start_idx = self.ptr
        self.count = 0

class TRPOBuffer(GAEBuffer):
    def __init__(self, obs_shape, ac_shape, size, gamma=0.99, lam=0.95):
        self.mu_buf = np.zeros((size,) + ac_shape, dtype=np.float32)
        self.logstd_buf = np.zeros((1,) + ac_shape, dtype=np.float32)
        super().__init__(obs_shape=obs_shape, ac_shape=ac_shape, size=size,
                         gamma=gamma, lam=lam)

    def store(self, obs, ac, rew, done, val, logp, mu, logstd):
        assert mu.shape == self.ac_shape
        assert logstd.shape == self.ac_shape
        self.mu_buf[self.ptr] = mu
        self.logstd_buf[0] = logstd
        super().store(obs, ac, rew, done, val, logp)

    def finish_path(self, last_val=None):
        super().finish_path(last_val=last_val)

    def get(self):
        assert self.ptr == self.max_size
        return [self.obs_buf,
                self.ac_buf,
                self.adv_buf,
                self.ret_buf,
                self.logp_buf,
                self.val_buf[:-1],
                self.mu_buf,
                self.logstd_buf]
