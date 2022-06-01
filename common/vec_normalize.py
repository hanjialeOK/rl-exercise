import numpy as np

from common.running_mean_std import RunningMeanStd


class VecNormalize():

    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., epsilon=1e-8):
        self.env = env
        self.ob_rms = RunningMeanStd(
            shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.epsilon = epsilon
        self.ac_low = self.env.action_space.low[0]
        self.ac_high = self.env.action_space.high[0]
        # self.obs_buf = []
        # self.rew_buf = []
        # self.done_buf = []

    def step(self, ac):
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, self.ac_low, self.ac_high)
        obs, rew, done, info = self.env.step(ac)
        self.raw_obs = obs.copy()
        self.raw_rew = rew
        # self.obs_buf.append(self.raw_obs)
        # self.rew_buf.append(self.raw_rew)
        # self.done_buf.append(done)
        obs = self._obfilt(obs)
        if self.ret_rms:
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon),
                          -self.cliprew, self.cliprew)
        return obs, rew, done, info

    def get_raw(self):
        return self.raw_obs, self.raw_rew

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var +
                          self.epsilon), -self.clipob, self.clipob)
        return obs

    def reset(self):
        obs = self.env.reset()
        return self._obfilt(obs)

    def update_rms(self, obs, ret):
        # obs_all = np.vstack(self.obs_buf).astype(np.float64)
        # rew_all = np.array(self.rew_buf).astype(np.float32)
        # done_all = np.array(self.done_buf).astype(np.float32)
        # ret_all = np.zeros_like(rew_all)
        # lastrew = 0.0
        # for t in reversed(range(done_all.shape[0])):
        #     nondone = 1 - done_all[t]
        #     ret_all[t] = rew_all[t] + 0.995 * nondone * lastrew
        #     lastrew = ret_all[t]

        if self.ob_rms:
            obs_raw = self._unnormalize_obs(obs)
            self.ob_rms.update(obs_raw)
            # self.ob_rms.update(obs_all)
        if self.ret_rms:
            ret_raw = self._unnormalize_ret(ret)
            self.ret_rms.update(ret_raw)
            # self.ret_rms.update(ret_all)

        # self.obs_buf = []
        # self.rew_buf = []
        # self.done_buf = []

    def _unnormalize_obs(self, obs):
        if self.ob_rms:
            return (obs * np.sqrt(self.ob_rms.var + self.epsilon)) + self.ob_rms.mean
        return obs

    def _unnormalize_ret(self, ret):
        if self.ret_rms:
            return ret * np.sqrt(self.ret_rms.var + self.epsilon)
        return ret

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def spec(self):
        return self.env.spec

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    def close(self):
        return self.env.close()

    def render(self, mode):
        return self.env.render(mode)
