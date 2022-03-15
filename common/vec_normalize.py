import numpy as np


class VecNormalize():
    """
    A wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10.,
                 gamma=0.99, epsilon=1e-8, use_tf=False):
        if use_tf:
            from common.running_mean_std import TfRunningMeanStd
            self.ob_rms = TfRunningMeanStd(
                shape=env.observation_space.shape, scope='ob_rms') if ob else None
            self.ret_rms = TfRunningMeanStd(
                shape=(), scope='ret_rms') if ret else None
        else:
            from common.running_mean_std import RunningMeanStd
            self.ob_rms = RunningMeanStd(
                shape=env.observation_space.shape) if ob else None
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.env = env
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = 0.
        self.raw_rew = 0.
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rew, new, info = self.env.step(action)
        self.raw_rew = rew
        self.ret = self.ret * self.gamma + rew
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret]))
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon),
                          -self.cliprew, self.cliprew)
        if new:
            self.ret = 0.
        return obs, rew, new, info

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = 0.
        obs = self.env.reset()
        return self._obfilt(obs)

    def get_original_reward(self):
        return self.raw_rew

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
