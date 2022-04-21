from common.vec_env.vec_env import VecEnvWrapper
import numpy as np


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10.,
                 gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        from common.running_mean_std import RunningMeanStd
        self.ob_rms = RunningMeanStd(
            shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        # self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        # self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            # self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var +
                           self.epsilon), -self.cliprew, self.cliprew)
        # self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            # self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var +
                          self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        # self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def update_rms(self, obs, ret):
        if self.ob_rms:
            obs_raw = self._unnormalize_obs(obs)
            self.ob_rms.update(obs_raw)
        if self.ret_rms:
            ret_raw = self._unnormalize_ret(ret)
            self.ret_rms.update(ret_raw)

    def _unnormalize_obs(self, obs):
        if self.ob_rms:
            return (obs * np.sqrt(self.ob_rms.var + self.epsilon)) + self.ob_rms.mean
        return obs

    def _unnormalize_ret(self, ret):
        if self.ret_rms:
            return ret * np.sqrt(self.ret_rms.var + self.epsilon)
        return ret
