import numpy as np


class VecNormalize():
    """
    A wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10.,
                 gamma=0.99, epsilon=1e-8, use_tf=False):
        if use_tf:
            raise NotImplementedError
        else:
            from common.running_mean_std import RunningMeanStd2
            self.ob_rms = RunningMeanStd2(
                shape=env.observation_space.shape) if ob else None
            self.ret_rms = RunningMeanStd2(shape=()) if ret else None
        self.env = env
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = 0.
        self.raw_rew = 0.
        self.gamma = gamma
        self.epsilon = epsilon
        self.ep_ret = 0.0
        self.ep_len = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # Episode info
        self.ep_ret += rew
        self.ep_len += 1
        # Discounted ret
        self.ret = self.ret * self.gamma + rew
        # Normalized obs
        obs = self._obfilt(obs)
        rew = self._rewfilt(rew)
        if done:
            epinfo = {'r': self.ep_ret, 'l': self.ep_len}
            info['episode'] = epinfo
            self.ep_ret = 0.
            self.ep_len = 0
            self.ret = 0.
        return obs, rew, done, info

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
        return obs

    def _rewfilt(self, rew):
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon),
                          -self.cliprew, self.cliprew)
        return rew

    def reset(self):
        self.ret = 0.
        self.ep_ret = 0.
        self.ep_len = 0
        obs = self.env.reset()
        return self._obfilt(obs)

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
