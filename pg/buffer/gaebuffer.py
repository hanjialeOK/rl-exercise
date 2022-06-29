import numpy as np
import scipy.signal


class GAEBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, ) + obs_dim, dtype=np.float64)
        self.act_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((size, ), dtype=np.float32)
        self.rew_buf = np.zeros((size, ), dtype=np.float32)
        self.done_buf = np.zeros((size, ), dtype=np.float32)
        self.ret_buf = np.zeros((size, ), dtype=np.float32)
        self.val_buf = np.zeros((size, ), dtype=np.float32)
        self.next_val_buf = np.zeros((size, ), dtype=np.float32)
        self.logp_buf = np.zeros((size, ), dtype=np.float32)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = size
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        assert obs.shape == self.obs_dim
        assert act.shape == self.act_dim
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_ob=None, last_val=None):
        start = self.path_start_idx
        self.next_val_buf[start:self.ptr-1] = self.val_buf[start+1:self.ptr]
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
        self.path_start_idx = 0

        return [self.obs_buf,
                self.act_buf,
                self.adv_buf,
                self.ret_buf,
                self.logp_buf,
                self.val_buf]

    def get_rms_data(self):
        assert self.ptr == self.max_size
        return [self.obs_buf,
                self.ret_buf]

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0


class GAEVBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, obs_dim, act_dim, size, nlatest=1, gamma=0.99, lam=0.95, uniform=False):
        max_size = size * nlatest
        self.obs_buf = np.zeros((max_size, ) + obs_dim, dtype=np.float64)
        self.obs2_buf = np.zeros((max_size, ) + obs_dim, dtype=np.float64)
        self.act_buf = np.zeros((max_size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((max_size, ), dtype=np.float32)
        self.rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.done_buf = np.zeros((max_size, ), dtype=np.float32)
        self.trun_buf = np.zeros((max_size, ), dtype=np.float32)
        self.ret_buf = np.zeros((max_size, ), dtype=np.float32)
        self.val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.next_val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.logp_buf = np.zeros((max_size, ), dtype=np.float32)
        self.logp_pik_buf = np.zeros((max_size, ), dtype=np.float32)
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.size = size
        self.nlatest = nlatest
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.count = 0
        self.uniform = uniform

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        assert obs.shape == self.obs_dim
        assert act.shape == self.act_dim
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        self.count = min(self.count+1, self.max_size)

    def finish_path(self, last_ob=None, last_val=None):
        start = self.path_start_idx
        self.trun_buf[start:self.ptr-1] = 0
        self.trun_buf[self.ptr-1] = 1
        self.next_val_buf[start:self.ptr-1] = self.val_buf[start+1:self.ptr]
        self.next_val_buf[self.ptr-1] = last_val
        self.obs2_buf[start:self.ptr-1] = self.obs_buf[start+1:self.ptr]
        self.obs2_buf[self.ptr-1] = last_ob

        # GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(self.path_start_idx, self.ptr)):
            nontruncated = 1.0 - self.trun_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = \
                delta + self.gamma * self.lam * nontruncated * lastgaelam
            lastgaelam = self.adv_buf[t]

        self.ret_buf[start:self.ptr] = self.adv_buf[start:self.ptr] + \
            self.val_buf[start:self.ptr]

        self.path_start_idx = self.ptr

    def vtrace(self, compute_v_pik, compute_logp_pik):
        assert self.ptr == self.size
        assert self.count % self.size == 0 and self.count > 0

        self.val_buf[:self.count] = compute_v_pik(self.obs_buf[:self.count])
        self.next_val_buf[:self.count] = compute_v_pik(self.obs2_buf[:self.count])
        self.logp_pik_buf[:self.count] = compute_logp_pik(
            self.obs_buf[:self.count], self.act_buf[:self.count])

        rho = np.exp(self.logp_pik_buf[:self.count] - self.logp_buf[:self.count])
        # Reduce bias here!
        rho = np.minimum(rho, 1.0)

        lastgaelam = 0.0
        for t in reversed(range(self.count)):
            nondone = 1.0 - self.done_buf[t]
            nontruncated = 1.0 - self.trun_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * nondone * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = delta + \
                self.gamma * self.lam * nontruncated * lastgaelam
            lastgaelam = rho[t] * self.adv_buf[t]
        self.ret_buf[:self.count] = self.adv_buf[:self.count] * rho + \
            self.val_buf[:self.count]

        M_active = self.count // self.size
        weights_active = self.weights[:M_active]
        weights_active = weights_active / np.sum(weights_active)
        weights_active *= M_active
        self.weights_all = np.repeat(weights_active, self.size)
        if self.uniform:
            self.weights_all = np.ones(self.count)

        # Reset ptr
        self.ptr = 0
        self.path_start_idx = 0

        return [self.obs_buf[:self.count],
                self.act_buf[:self.count],
                self.adv_buf[:self.count],
                self.ret_buf[:self.count],
                self.logp_buf[:self.count],
                self.val_buf[:self.count],
                self.logp_pik_buf[:self.count],
                self.weights_all]

    def update(self):
        index1 = 1 * self.size
        index2 = (self.nlatest-1) * self.size
        self.obs_buf[index1:] = self.obs_buf[:index2]
        self.obs2_buf[index1:] = self.obs2_buf[:index2]
        self.act_buf[index1:] = self.act_buf[:index2]
        self.adv_buf[index1:] = self.adv_buf[:index2]
        self.rew_buf[index1:] = self.rew_buf[:index2]
        self.done_buf[index1:] = self.done_buf[:index2]
        self.trun_buf[index1:] = self.trun_buf[:index2]
        self.ret_buf[index1:] = self.ret_buf[:index2]
        self.val_buf[index1:] = self.val_buf[:index2]
        self.next_val_buf[index1:] = self.next_val_buf[:index2]
        self.logp_buf[index1:] = self.logp_buf[:index2]
        self.logp_pik_buf[index1:] = self.logp_pik_buf[:index2]

    # def get_obs(self):
    #     assert self.count % self.size == 0 and self.count > 0
    #     return [self.obs_buf[:self.count],
    #             self.obs2_buf[:self.count],
    #             self.act_buf[:self.count]]

    def get_rms_data(self):
        assert self.ptr == self.size
        # Return the latest RMS data
        return [self.obs_buf[:self.ptr],
                self.ret_buf[:self.ptr]]

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0
        self.count = 0


class DISCBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, obs_dim, act_dim, size, nlatest=1, gamma=0.99, lam=0.95, uniform=False):
        max_size = size * nlatest
        self.obs_buf = np.zeros((max_size, ) + obs_dim, dtype=np.float64)
        self.obs2_buf = np.zeros((max_size, ) + obs_dim, dtype=np.float64)
        self.act_buf = np.zeros((max_size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((max_size, ), dtype=np.float32)
        self.rew_buf = np.zeros((max_size, ), dtype=np.float32)
        self.done_buf = np.zeros((max_size, ), dtype=np.float32)
        self.trun_buf = np.zeros((max_size, ), dtype=np.float32)
        self.ret_buf = np.zeros((max_size, ), dtype=np.float32)
        self.val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.next_val_buf = np.zeros((max_size, ), dtype=np.float32)
        self.logp_disc_buf = np.zeros((max_size, ) + act_dim, dtype=np.float32)
        self.logp_disc_pik_buf = np.zeros((max_size, ) + act_dim, dtype=np.float32)
        self.weights = np.array([0.4, 0.3, 0.2, 0.1])
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.size = size
        self.nlatest = nlatest
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.count = 0
        self.uniform = uniform

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        assert obs.shape == self.obs_dim
        assert act.shape == self.act_dim
        assert logp.shape == self.act_dim
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_disc_buf[self.ptr] = logp
        self.ptr += 1
        self.count = min(self.count+1, self.max_size)

    def finish_path(self, last_ob=None, last_val=None):
        start = self.path_start_idx
        self.trun_buf[start:self.ptr-1] = 0
        self.trun_buf[self.ptr-1] = 1
        self.next_val_buf[start:self.ptr-1] = self.val_buf[start+1:self.ptr]
        self.next_val_buf[self.ptr-1] = last_val
        self.obs2_buf[start:self.ptr-1] = self.obs_buf[start+1:self.ptr]
        self.obs2_buf[self.ptr-1] = last_ob

        # GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(self.path_start_idx, self.ptr)):
            nontruncated = 1.0 - self.trun_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = \
                delta + self.gamma * self.lam * nontruncated * lastgaelam
            lastgaelam = self.adv_buf[t]

        self.ret_buf[start:self.ptr] = self.adv_buf[start:self.ptr] + \
            self.val_buf[start:self.ptr]

        self.path_start_idx = self.ptr

    def vtrace(self, compute_v_pik, compute_logp_pik):
        assert self.ptr == self.size
        assert self.count % self.size == 0 and self.count > 0

        self.val_buf[:self.count] = compute_v_pik(self.obs_buf[:self.count])
        self.next_val_buf[:self.count] = compute_v_pik(self.obs2_buf[:self.count])
        self.logp_disc_pik_buf[:self.count] = compute_logp_pik(
            self.obs_buf[:self.count], self.act_buf[:self.count])

        logp_pik = np.sum(self.logp_disc_pik_buf[:self.count], axis=1)
        logp_a = np.sum(self.logp_disc_buf[:self.count], axis=1)

        rho = np.exp(logp_pik - logp_a)
        # Reduce bias here!
        rho = np.minimum(rho, 1.0)

        lastgaelam = 0.0
        for t in reversed(range(self.count)):
            nondone = 1.0 - self.done_buf[t]
            nontruncated = 1.0 - self.trun_buf[t]
            delta = self.rew_buf[t] + \
                self.gamma * nondone * self.next_val_buf[t] - self.val_buf[t]
            self.adv_buf[t] = delta + \
                self.gamma * self.lam * nontruncated * lastgaelam
            lastgaelam = rho[t] * self.adv_buf[t]
        self.ret_buf[:self.count] = self.adv_buf[:self.count] * rho + \
            self.val_buf[:self.count]

        M_active = self.count // self.size
        weights_active = self.weights[:M_active]
        weights_active = weights_active / np.sum(weights_active)
        weights_active *= M_active
        self.weights_all = np.repeat(weights_active, self.size)
        if self.uniform:
            self.weights_all = np.ones(self.count)

        # Reset ptr
        self.ptr = 0
        self.path_start_idx = 0

        return [self.obs_buf[:self.count],
                self.act_buf[:self.count],
                self.adv_buf[:self.count],
                self.ret_buf[:self.count],
                self.logp_disc_buf[:self.count],
                self.val_buf[:self.count],
                self.logp_disc_pik_buf[:self.count],
                self.weights_all]

    def update(self):
        index1 = 1 * self.size
        index2 = (self.nlatest-1) * self.size
        self.obs_buf[index1:] = self.obs_buf[:index2]
        self.obs2_buf[index1:] = self.obs2_buf[:index2]
        self.act_buf[index1:] = self.act_buf[:index2]
        self.adv_buf[index1:] = self.adv_buf[:index2]
        self.rew_buf[index1:] = self.rew_buf[:index2]
        self.done_buf[index1:] = self.done_buf[:index2]
        self.trun_buf[index1:] = self.trun_buf[:index2]
        self.ret_buf[index1:] = self.ret_buf[:index2]
        self.val_buf[index1:] = self.val_buf[:index2]
        self.next_val_buf[index1:] = self.next_val_buf[:index2]
        self.logp_disc_buf[index1:] = self.logp_disc_buf[:index2]
        self.logp_disc_pik_buf[index1:] = self.logp_disc_pik_buf[:index2]

    # def get_obs(self):
    #     assert self.count % self.size == 0 and self.count > 0
    #     return [self.obs_buf[:self.count],
    #             self.obs2_buf[:self.count],
    #             self.act_buf[:self.count]]

    def get_rms_data(self):
        assert self.ptr == self.size
        # Return the latest RMS data
        return [self.obs_buf[:self.ptr],
                self.ret_buf[:self.ptr]]

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0
        self.count = 0

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
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.mu_buf = np.zeros((size,) + act_dim, dtype=np.float32)
        self.logstd_buf = np.zeros((1,) + act_dim, dtype=np.float32)
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, size=size,
                         gamma=gamma, lam=lam)

    def store(self, obs, act, rew, done, val, logp, mu, logstd):
        assert mu.shape == self.act_dim
        assert logstd.shape == self.act_dim
        self.mu_buf[self.ptr] = mu
        self.logstd_buf[0] = logstd
        super().store(obs, act, rew, done, val, logp)

    def finish_path(self, last_val=None):
        super().finish_path(last_val=last_val)

    def get(self):
        assert self.ptr == self.max_size
        return [self.obs_buf,
                self.act_buf,
                self.adv_buf,
                self.ret_buf,
                self.logp_buf,
                self.val_buf[:-1],
                self.mu_buf,
                self.logstd_buf]
