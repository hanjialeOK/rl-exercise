from common.vec_env.vec_env import VecEnvWrapper
import numpy as np


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        from common.running_mean_std import RunningMeanStd
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.raw_obs = obs.copy()
        self.raw_rews = rews.copy()
        self.ret = self.ret * self.gamma + rews
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = self._rewfilt(rews)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def _rewfilt(self, rews):
        if self.ret_rms:
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon),
                           -self.cliprew, self.cliprew)
            return rews
        else:
            return rews

    def get_raw(self):
        return self.raw_obs, self.raw_rews

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        self.raw_obs = obs.copy()
        self.raw_rews = 0
        return self._obfilt(obs)
