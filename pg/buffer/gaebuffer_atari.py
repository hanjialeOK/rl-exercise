import numpy as np

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.reshape(s[0] * s[1], *s[2:])


class GAEBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, env, nenv, horizon, gamma=0.99, lam=0.95, compute_v=None):
        obs_shape = (84, 84, 4)
        ac_shape = env.action_space.shape
        obs_dtype = env.observation_space.dtype.name
        ac_dtype = env.action_space.dtype.name
        self.obs_buf = np.zeros((horizon, nenv) + obs_shape, dtype=obs_dtype)
        self.obs2_buf = np.zeros((horizon, nenv) + obs_shape, dtype=obs_dtype)
        self.ac_buf = np.zeros((horizon, nenv) + ac_shape, dtype=ac_dtype)
        self.rew_buf = np.zeros((horizon, nenv), dtype=np.float32)
        self.done_buf = np.zeros((horizon, nenv), dtype=np.float32)
        self.ret_buf = np.zeros((horizon, nenv), dtype=np.float32)
        self.adv_buf = np.zeros((horizon, nenv), dtype=np.float32)
        self.val_buf = np.zeros((horizon, nenv), dtype=np.float32)
        self.next_val_buf = np.zeros((horizon, nenv), dtype=np.float32)
        self.neglogp_buf = np.zeros((horizon, nenv), dtype=np.float32)
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.max_size = horizon
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.compute_v = compute_v

    def store(self, obs=None, ac=None, rew=None, done=None, obs2=None, val=None, neglogp=None, **kwargs):
        assert self.ptr < self.max_size
        # assert obs.shape == self.obs_shape
        # assert ac.shape == self.ac_shape
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.ac_buf[self.ptr] = ac
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.neglogp_buf[self.ptr] = neglogp
        self.ptr += 1

    def get(self):
        assert self.ptr == self.max_size

        self.next_val_buf[:self.ptr-1] = self.val_buf[1:self.ptr]

        last_ob2 = self.obs2_buf[self.ptr-1]
        self.next_val_buf[self.ptr-1] = self.compute_v(last_ob2)

        # GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(0, self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * nondone * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            lastgaelam = self.adv_buf[t]

        self.ret_buf[:] = self.adv_buf + self.val_buf

        # Reset ptr
        self.ptr = 0

        return [sf01(self.obs_buf),
                sf01(self.ac_buf),
                sf01(self.adv_buf),
                sf01(self.ret_buf),
                sf01(self.val_buf),
                sf01(self.neglogp_buf)]

    def reset(self):
        self.ptr = 0


class GAEVBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, env, nenv, horizon, nlatest=1, gamma=0.99, lam=0.95,
                 compute_v_pik=None, compute_neglogp_pik=None):
        obs_shape = (84, 84, 4)
        ac_shape = env.action_space.shape
        obs_dtype = env.observation_space.dtype.name
        ac_dtype = env.action_space.dtype.name
        max_size = horizon * nlatest
        self.obs_buf = np.zeros((max_size, nenv) + obs_shape, dtype=obs_dtype)
        self.obs2_buf = np.zeros((max_size, nenv) + obs_shape, dtype=obs_dtype)
        self.ac_buf = np.zeros((max_size, nenv) + ac_shape, dtype=ac_dtype)
        self.rew_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.done_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.ret_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.adv_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.val_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.next_val_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.neglogp_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.neglogp_pik_buf = np.zeros((max_size, nenv), dtype=np.float32)
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.max_size = max_size
        self.horizon = horizon
        self.nlatest = nlatest
        self.nenv = nenv
        self.nbatch = horizon * nenv
        self.gamma = gamma
        self.lam = lam
        self.ptr = self.max_size - self.horizon
        self.count = 0
        self.compute_v_pik = compute_v_pik
        self.compute_neglogp_pik = compute_neglogp_pik

    def store(self, obs=None, ac=None, rew=None, done=None, obs2=None, val=None, neglogp=None, **kwargs):
        assert self.ptr < self.max_size
        # assert obs.shape == self.obs_shape
        # assert raw_obs.shape == self.obs_shape
        # assert ac.shape == self.ac_shape
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.ac_buf[self.ptr] = ac
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.neglogp_buf[self.ptr] = neglogp
        self.ptr += 1
        self.count = min(self.count+1, self.max_size)

    def vtrace(self):
        assert self.ptr == self.max_size
        assert self.count % self.horizon == 0 and self.count > 0

        # obs_all = self.obs_buf.reshape((self.max_size * self.nenv, ) + self.obs_shape)
        for i in range(self.nlatest):
            # batch_inds = slice(i*self.nbatch, (i+1)*self.nbatch)
            mbinds = slice(i*self.horizon, (i+1)*self.horizon)
            obs_batch = self.obs_buf[mbinds].reshape((self.horizon * self.nenv, ) + self.obs_shape)
            val_batch = self.compute_v_pik(obs_batch)
            self.val_buf[mbinds] = val_batch.reshape((self.horizon, self.nenv))

        # obs2_all = self.obs2_buf.reshape((self.max_size * self.nenv, ) + self.obs_shape)
        # for i in range(self.nlatest):
            # batch_inds = slice(i*self.nbatch, (i+1)*self.nbatch)
            # mbinds = slice(i*self.horizon, (i+1)*self.horizon)
            obs2_batch = self.obs2_buf[mbinds].reshape((self.horizon * self.nenv, ) + self.obs_shape)
            next_val_batch = self.compute_v_pik(obs2_batch)
            self.next_val_buf[mbinds] = next_val_batch.reshape((self.horizon, self.nenv))

        # ac_all = self.ac_buf.reshape((self.max_size * self.nenv))
        # for i in range(self.nlatest):
            # batch_inds = slice(i*self.nbatch, (i+1)*self.nbatch)
            # mbinds = slice(i*self.horizon, (i+1)*self.horizon)
            ac_batch = self.ac_buf[mbinds].reshape((self.horizon * self.nenv))
            neglogp_pik_batch = self.compute_neglogp_pik(obs_batch, ac_batch)
            self.neglogp_pik_buf[mbinds] = neglogp_pik_batch.reshape((self.horizon, self.nenv))

        rho = np.exp(self.neglogp_buf - self.neglogp_pik_buf)
        # Reduce bias here!
        rho = np.minimum(rho, 1.0)

        lastgaelam = 0.0
        for t in reversed(range(self.max_size - self.count, self.max_size)):
            nondone = 1.0 - self.done_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * nondone * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            lastgaelam = rho[t] * self.adv_buf[t]
        self.ret_buf[:] = self.adv_buf * rho + self.val_buf

        # Reset ptr
        self.ptr = self.max_size - self.horizon

        return [sf01(self.obs_buf[-self.count:]),
                sf01(self.ac_buf[-self.count:]),
                sf01(self.adv_buf[-self.count:]),
                sf01(self.ret_buf[-self.count:]),
                sf01(self.val_buf[-self.count:]),
                sf01(self.neglogp_buf[-self.count:]),
                sf01(self.neglogp_pik_buf[-self.count:])]

    def update(self):
        tail = self.max_size - self.horizon
        head = self.horizon
        self.obs_buf[:tail] = self.obs_buf[head:]
        self.obs2_buf[:tail] = self.obs2_buf[head:]
        self.rew_buf[:tail] = self.rew_buf[head:]
        self.ac_buf[:tail] = self.ac_buf[head:]
        self.done_buf[:tail] = self.done_buf[head:]
        # self.ret_buf[:tail] = self.ret_buf[head:]
        # self.adv_buf[:tail] = self.adv_buf[head:]
        # self.val_buf[:tail] = self.val_buf[head:]
        # self.next_val_buf[:tail] = self.next_val_buf[head:]
        self.neglogp_buf[:tail] = self.neglogp_buf[head:]
        # self.neglogp_pik_buf[:tail] = self.neglogp_pik_buf[head:]

    def reset(self):
        self.ptr = self.max_size - self.horizon
        self.count = 0


class ACERBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, env, horizon, replay_size=5000, gamma=0.99,
                 compute_v=None, compute_sdn_pik=None):
        obs_shape = env.observation_space.shape
        ac_shape = env.action_space.shape
        obs_dtype = env.observation_space.dtype.name
        ac_dtype = env.action_space.dtype.name
        max_size = replay_size
        self.nlatest = replay_size // horizon
        self.obs_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.obs2_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.raw_obs_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.raw_obs2_buf = np.zeros((max_size, ) + obs_shape, dtype=obs_dtype)
        self.ac_buf = np.zeros((max_size, ) + ac_shape, dtype=ac_dtype)
        self.rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.raw_rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.done_buf = np.zeros((max_size, ), dtype=np.float32)
        self.ret_buf = np.zeros((max_size, ), dtype=np.float32)
        self.adv_buf = np.zeros((max_size, ), dtype=np.float32)
        self.val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.qval_buf = np.zeros((max_size, ), dtype=np.float32)
        self.qret_buf = np.zeros((max_size, ), dtype=np.float32)
        self.qopc_buf = np.zeros((max_size, ), dtype=np.float32)
        self.next_val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.neglogp_buf = np.zeros((max_size, ), dtype=np.float32)
        self.neglogp_pik_buf = np.zeros((max_size, ), dtype=np.float32)
        self.mean_buf = np.zeros((max_size, ) + ac_shape, dtype=np.float32)
        self.logstd_buf = np.zeros((max_size, ) + ac_shape, dtype=np.float32)
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.max_size = max_size
        self.horizon = horizon
        self.gamma = gamma
        self.ptr = 0
        self.count = 0
        self.obfilt = env._obfilt
        self.rewfilt = env._rewfilt
        self.compute_v = compute_v
        self.compute_sdn_pik = compute_sdn_pik
        self.n_trajs = 0

    def store(self, obs=None, ac=None, rew=None, done=None, obs2=None, val=None, neglogp=None,
              raw_obs=None, raw_rew=None, raw_obs2=None, mean=None, logstd=None):
        assert self.ptr < self.max_size
        # assert obs.shape == self.obs_shape
        # assert raw_obs.shape == self.obs_shape
        # assert ac.shape == self.ac_shape
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = obs2
        self.ac_buf[self.ptr] = ac
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.raw_obs_buf[self.ptr] = raw_obs
        self.raw_obs2_buf[self.ptr] = raw_obs2
        self.raw_rew_buf[self.ptr] = raw_rew
        self.val_buf[self.ptr] = val
        self.neglogp_buf[self.ptr] = neglogp
        self.mean_buf[self.ptr] = mean
        self.logstd_buf[self.ptr] = logstd
        self.ptr = (self.ptr + 1) % self.max_size
        self.count = min(self.count+1, self.max_size)

    def finish_path(self):
        self.n_trajs = min(self.nlatest, self.n_trajs + 1)

    def get(self, on_policy=False):
        assert self.ptr % self.horizon == 0
        assert self.count % self.horizon == 0 and self.count > 0

        if self.obfilt:
            self.obs_buf[:] = self.obfilt(self.raw_obs_buf)
            self.obs2_buf[:] = self.obfilt(self.raw_obs2_buf)
        if self.rewfilt:
            self.rew_buf[:] = self.rewfilt(self.raw_rew_buf)

        if on_policy:
            pos = self.nlatest - 1 if self.ptr == 0 else self.ptr // self.horizon - 1
        else:
            pos = np.random.randint(0, self.count // self.horizon)
        start = pos * self.horizon
        end = (pos + 1) * self.horizon

        last_ob = self.obs2_buf[end-1].reshape(1, -1)
        obs = np.append(self.obs_buf[start:end], last_ob, axis=0)

        return [obs,
                self.ac_buf[start:end],
                self.rew_buf[start:end],
                self.done_buf[start:end],
                self.neglogp_buf[start:end],
                self.mean_buf[start:end],
                self.logstd_buf[start:end]]

    def get_rms_data(self):
        raise NotImplementedError
        assert self.ptr == self.max_size
        # Return the latest RMS data
        return [self.obs_buf[-self.horizon:],
                self.ret_buf[-self.horizon:]]

    def reset(self):
        self.ptr = 0
        self.count = 0
        self.n_trajs = 0