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
        self.path_start_idx = self.ptr

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
        max_size = horizon * nlatest // nenv
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

        obs_all = self.obs_buf.reshape((self.max_size * self.nenv, ) + self.obs_shape)
        val_all = self.compute_v_pik(obs_all)
        self.val_buf[:] = val_all.reshape((self.max_size, self.nenv))
        
        obs2_all = self.obs2_buf.reshape((self.max_size * self.nenv, ) + self.obs_shape)
        next_val_all = self.compute_v_pik(obs2_all)
        self.next_val_buf[:] = next_val_all.reshape((self.max_size, self.nenv))

        ac_all = self.ac_buf.reshape((self.max_size * self.nenv))
        neglogp_pik_all = self.compute_neglogp_pik(obs_all, ac_all)
        self.neglogp_pik_buf[:] = neglogp_pik_all.reshape((self.max_size, self.nenv))

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
        self.path_start_idx = self.ptr

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
