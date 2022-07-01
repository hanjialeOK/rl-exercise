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
        self.raw_obs = np.zeros(self.observation_space.shape)
        self.raw_rew = 0.

    def step(self, ac):
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, self.ac_low, self.ac_high)
        obs, rew, done, info = self.env.step(ac)
        self.raw_obs[:] = self.obs
        self.raw_rew = rew
        obs = self._obfilt(obs)
        rew = self._rewfilt(rew)
        return obs, rew, done, info

    def get_raw(self):
        return self.raw_obs, self.raw_rew

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), 
                          -self.clipob, self.clipob)
        return obs

    def _rewfilt(self, rew):
        if self.ret_rms:
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon),
                          -self.cliprew, self.cliprew)
        return rew

    def reset(self):
        obs = self.env.reset()
        self.raw_obs[:] = obs
        self.raw_rew = 0.
        return self._obfilt(obs)

    def update_rms(self, obs, ret):
        if self.ob_rms:
            obs_raw = self._obunfilt(obs)
            self.ob_rms.update(obs_raw)
        if self.ret_rms:
            ret_raw = self._rewunfilt(ret)
            self.ret_rms.update(ret_raw)

    def _obunfilt(self, obs):
        if self.ob_rms:
            return (obs * np.sqrt(self.ob_rms.var + self.epsilon)) + self.ob_rms.mean
        return obs

    def _rewunfilt(self, rew):
        if self.ret_rms:
            return rew * np.sqrt(self.ret_rms.var + self.epsilon)
        return rew

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

class VecNormalize2():

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
        self.gamma = 0.99
        self.ret = 0.
        self.raw_obs = np.zeros(self.observation_space.shape)
        self.raw_rew = 0.

    def step(self, ac):
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, self.ac_low, self.ac_high)
        obs, rew, done, info = self.env.step(ac)
        self.raw_obs[:] = obs
        self.raw_rew = rew

        self.ob_rms.update(np.reshape(obs, (1, -1)))
        obs = self._obfilt(obs)

        self.ret = self.ret * self.gamma + rew
        self.ret_rms.update(np.array([self.ret]))
        rew = self._rewfilt(rew)
        if done:
            self.ret = 0.

        return obs, rew, done, info

    def get_raw(self):
        return self.raw_obs, self.raw_rew

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), 
                          -self.clipob, self.clipob)
        return obs

    def _rewfilt(self, rew):
        if self.ret_rms:
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon),
                          -self.cliprew, self.cliprew)
        return rew

    def reset(self):
        self.ret = 0.
        obs = self.env.reset()
        self.raw_obs[:] = obs
        self.raw_rew = 0.
        return self._obfilt(obs)

    def update_rms(self, obs=None, ret=None):
        pass

    def _obunfilt(self, obs):
        if self.ob_rms:
            return (obs * np.sqrt(self.ob_rms.var + self.epsilon)) + self.ob_rms.mean
        return obs

    def _rewunfilt(self, ret):
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
