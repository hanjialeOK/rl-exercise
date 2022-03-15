import numpy as np
import random
import cv2
import gym

from common.serialization_utils import convert_json


def create_atari_environment(game_name=None):
    """
    Args:
        game_name: str, the name of the Atari 2600 domain.
        sticky_actions: bool, whether to use sticky_actions as per Machado et al.
    Returns:
        An Atari 2600 environment with some standard preprocessing.
    """
    env = gym.make(game_name)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    env = AtariPreprocessing(env)
    return env


class AtariPreprocessing(object):
    """A class implementing image preprocessing for Atari 2600 agents.
        * Frame skipping (defaults to 4).
        * Terminal signal when a life is lost (off by default).
        * Grayscale and max-pooling of the last two frames.
        * Downsample the screen to a square image (defaults to 84x84).
    """

    def __init__(self, env, frame_skip=4, punish_on_loss=True, noop_max=30):
        self.config = convert_json(locals())

        assert env.get_action_meanings()[0] == 'NOOP'
        assert env.get_action_meanings()[1] == 'FIRE'
        assert len(env.get_action_meanings()) >= 3
        self._noop_action = 0
        self._env = env
        self._punish_on_loss = punish_on_loss
        self._frame_skip = frame_skip
        self._noop_max = noop_max
        self._width = 84
        self._height = 84

        # Stores temporary observations used for pooling over two successive
        # frames.
        obs_dims = env.observation_space.shape
        self._obs_buffer = np.zeros(
            (2,) + (obs_dims[0], obs_dims[1]), dtype=np.uint8)

        self._lives = 0
        # Unused but necessary.
        self.was_life_loss = False

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def spec(self):
        return self._env.spec

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def reward_range(self):
        return self._env.reward_range

    @property
    def metadata(self):
        return self._env.metadata

    def close(self):
        return self._env.close()

    def render(self, mode):
        return self._env.render(mode)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self._env.reset()
        # Reset recent raw observations.
        self._obs_buffer.fill(0)
        # Noop actions
        noops = random.randint(1, self._noop_max)
        obs = None
        for i in range(noops):
            _, _, done, _ = self._env.step(self._noop_action)
            # Record the last two frames.
            if i == noops - 2:
                self._env.ale.getScreenGrayscale(self._obs_buffer[0])
            if i == noops - 1:
                self._env.ale.getScreenGrayscale(self._obs_buffer[1])
            if done:
                self._env.reset()
        self._lives = self._env.ale.lives()
        obs = self._get_processed_obs()
        return obs

    def step(self, action):
        """Applies the given action in the environment.
        Remarks:
        * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
        * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.
        """
        accumulated_reward = 0.0

        for i in range(self._frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, terminal, info = self._env.step(action)
            accumulated_reward += reward

            # Negetive reward on life loss.
            lives = self._env.ale.lives()
            die = (lives < self._lives)
            self._lives = lives
            if self._punish_on_loss and die:
                accumulated_reward -= 1.0
            # We max-pool over the last two frames, in grayscale.
            if i == self._frame_skip - 2:
                self._env.ale.getScreenGrayscale(self._obs_buffer[0])
            if i == self._frame_skip - 1:
                self._env.ale.getScreenGrayscale(self._obs_buffer[1])
            if terminal:
                break

        # Pool the last two observations.
        observation = self._get_processed_obs()

        return observation, accumulated_reward, terminal, info

    def _get_processed_obs(self):
        # For efficiency, done in-place.
        np.maximum(self._obs_buffer[0], self._obs_buffer[1],
                   out=self._obs_buffer[0])

        obs = cv2.resize(
            self._obs_buffer[0], (self._width, self._height), interpolation=cv2.INTER_AREA)
        int_image = np.asarray(obs, dtype=np.uint8)
        return int_image
