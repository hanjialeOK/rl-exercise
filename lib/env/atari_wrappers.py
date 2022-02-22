import numpy as np
import random
import cv2
import gym
from gym.spaces.box import Box


def create_atari_environment(game_name=None):
    env = gym.make(game_name)
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    return env


class AtariWrapper(gym.Wrapper):
    """
    * Max 30 Noop actions.
    * Frame skipping (defaults to 4).
    * Process the last two frames.
    * Downsample the screen (defaults to 84x84).
    * Reduce reward on life loss.
    """

    def __init__(self, env, frame_skip=4, noop_max=30):
        """Return only every `frame_skip`-th frame"""
        super(AtariWrapper, self).__init__(env)
        # Make sure NOOP = 0, FIRE = 1.
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        self._noop_action = 0
        self._fire_action = 1
        self._frame_skip = frame_skip
        self._noop_max = noop_max
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._width = 84
        self._height = 84
        self._lives = 0

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._frame_skip):
            obs, reward, done, info = self.env.step(action)
            # Losing a life reduces the reward by one
            lives = self.env.unwrapped.ale.lives()
            die = (lives < self._lives)
            if die:
                total_reward -= 1
            self._lives = lives
            # Record the last two frames.
            if i == self._frame_skip - 2:
                self._obs_buffer[0] = obs
            if i == self._frame_skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        frame = self._get_processed_obs()

        return frame, total_reward, done, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        # Reset recent raw observations.
        self._obs_buffer.fill(0)
        # Take action on reset for environments that are fixed until firing.
        obs, _, done, _ = self.env.step(self._fire_action)
        # Noop actions
        noops = random.randint(1, self._noop_max)
        obs = None
        for i in range(noops):
            obs, _, done, _ = self.env.step(self._noop_action)
            # Record the last two frames.
            if i == noops - 2:
                self._obs_buffer[0] = obs
            if i == noops - 1:
                self._obs_buffer[1] = obs
            if done:
                obs = self.env.reset(**kwargs)
        # Record the initial lives.
        self._lives = self.env.unwrapped.ale.lives()
        obs = self._get_processed_obs()
        return obs

    def _get_processed_obs(self):
        max_frame = self._obs_buffer.max(axis=0)
        frame = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame


class EpisodicLifeEnv(gym.Wrapper):
    """
    * Max 30 Noop actions.
    * Frame skipping (defaults to 4).
    * Process the last two frames.
    * Downsample the screen (defaults to 84x84).
    * Terminal on life loss.
    """

    def __init__(self, env, frame_skip=4, noop_max=30):
        """Return only every `frame_skip`-th frame"""
        super(EpisodicLifeEnv, self).__init__(env)
        # Make sure NOOP = 0, FIRE = 1.
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        self._noop_action = 0
        self._fire_action = 1
        self._frame_skip = frame_skip
        self._noop_max = noop_max
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8)
        self._width = 84
        self._height = 84
        self._lives = 0
        self.was_real_done = True

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._frame_skip):
            obs, reward, done, info = self.env.step(action)

            self.was_real_done = done

            lives = self.env.unwrapped.ale.lives()
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            die = (lives < self._lives and lives > 0)
            self._lives = lives
            # Record the last two frames.
            if i == self._frame_skip - 2:
                self._obs_buffer[0] = obs
            if i == self._frame_skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if die or done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter

        frame = self._get_processed_obs()

        return frame, total_reward, die, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        # Reset recent raw observations.
        self._obs_buffer.fill(0)
        if self.was_real_done:
            self.env.reset(**kwargs)
            # Take action on reset for environments that are fixed until firing.
            obs, _, done, _ = self.env.step(self._fire_action)
            # Noop actions
            noops = random.randint(1, self._noop_max)
            obs = None
            for i in range(noops):
                obs, _, done, _ = self.env.step(self._noop_action)
                # Record the last two frames.
                if i == noops - 2:
                    self._obs_buffer[0] = obs
                if i == noops - 1:
                    self._obs_buffer[1] = obs
                if done:
                    obs = self.env.reset(**kwargs)
            # Record the initial lives.
            self._lives = self.env.unwrapped.ale.lives()
        else:
            # Take action on reset for environments that are fixed until firing.
            # Meanwhile, fire_action helps step out of old state.
            obs, _, done, _ = self.env.step(self._fire_action)
            self._obs_buffer[1] = obs
        obs = self._get_processed_obs()
        return obs

    def _get_processed_obs(self):
        max_frame = self._obs_buffer.max(axis=0)
        frame = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame
