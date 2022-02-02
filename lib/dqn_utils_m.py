import numpy as np
import random
import cv2
import gym
from gym.spaces.box import Box
import json

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def json_serializable(x):
	assert isinstance(x, dict)
	black_list = []
	for key in x.keys():
		if not is_jsonable(x[key]):
			black_list.append(key)
	for key in black_list:
		x.pop(key)
	return x

class History:
    def __init__(self):
        self.history = np.zeros(shape=(4, 84, 84), dtype=np.uint8)

    def add(self, screen):
        """
        screen: (84, 84)
        """
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def get(self):
        """
        return: (1, 4, 84, 84)
        """
        return np.expand_dims(self.history, axis=0)

    def reset(self):
        self.history *= 0

class ReplayMemory:
	def __init__(self, memory_size, batch_size):
		self.memory_size = memory_size
		self.actions = np.empty(self.memory_size, dtype = np.uint8)
		self.screens = np.empty((self.memory_size, 84, 84), dtype = np.uint8)
		self.rewards = np.empty(self.memory_size, dtype = np.int32)
		self.terminals = np.empty(self.memory_size, dtype = np.bool)
		self.history_length = 4
		self.dims = (84, 84)
		self.batch_size = batch_size
		self.count = 0
		self.current = 0

		# pre-allocate states and next_states for minibatch
		self.states = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
		self.next_states = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)

	def add(self, action, screen, reward, terminal):
		assert screen.shape == self.dims
		# NB! screen is next_state, after action and reward
		self.actions[self.current] = action
		self.screens[self.current, ...] = screen
		self.rewards[self.current] = reward
		self.terminals[self.current] = terminal
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.memory_size

	def sample(self):
		"""
		This process looks like:
			...
			<a0, s1, r1, t1>
			<a1, s2, r2, t2>
			<a2, s3, r3, t3>
			<a3, s4, r4, t4>
			<a4, s5, r5, t5>  <=  index
			...
		Returns:
			state = [s1, s2, s3, s4]
			next_state = [s2, s3, s4, s5]
			action = a4
			reward = r5
			terminal = t5
		Note: if t5 is True, s5 will be a bad observation. However, 
		      target = r5 + gamma * (1 - t5) * q_max(s5) = r5, which has no business with s5.
		"""
		assert self.count > self.history_length
		indexes = []
		while len(indexes) < self.batch_size:
			while True:
				index = random.randint(self.history_length, self.count - 1)
				# if wraps over current pointer, then get new one
				if index >= self.current and index - self.history_length < self.current:
					continue
				# if wraps over episode end, then get new one
				# Note: t3 is allowed to be True.
				if self.terminals[(index - self.history_length):index].any():
					continue
				# otherwise use this index
				break
			# NB! having index first is fastest in C-order matrices
			self.states[len(indexes), ...] = \
				self.screens[(index - self.history_length):index, ...]
			self.next_states[len(indexes), ...] = \
				self.screens[(index - (self.history_length - 1)):(index + 1), ...]
			indexes.append(index)
		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]

		return self.states, actions, rewards, self.next_states, terminals

def create_atari_environment(game_name=None, sticky_actions=True):
	"""Wraps an Atari 2600 Gym environment with some basic preprocessing.
	This preprocessing matches the guidelines proposed in Machado et al. (2017),
	"Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
	Problems for General Agents".
	The created environment is the Gym wrapper around the Arcade Learning
	Environment.
	The main choice available to the user is whether to use sticky actions or not.
	Sticky actions, as prescribed by Machado et al., cause actions to persist
	with some probability (0.25) when a new command is sent to the ALE. This
	can be viewed as introducing a mild form of stochasticity in the environment.
	We use them by default.
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
	Specifically, this provides the following subset from the JAIR paper
	(Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):
		* Frame skipping (defaults to 4).
		* Terminal signal when a life is lost (off by default).
		* Grayscale and max-pooling of the last two frames.
		* Downsample the screen to a square image (defaults to 84x84).
	More generally, this class follows the preprocessing guidelines set down in
	Machado et al. (2018), "Revisiting the Arcade Learning Environment:
	Evaluation Protocols and Open Problems for General Agents".
	"""

	def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
				 screen_size=84):
		"""Constructor for an Atari 2600 preprocessor.
		Args:
			environment: Gym environment whose observations are preprocessed.
			frame_skip: int, the frequency at which the agent experiences the game.
			terminal_on_life_loss: bool, If True, the step() method returns
				is_terminal=True whenever a life is lost. See Mnih et al. 2015.
			screen_size: int, size of a resized Atari 2600 frame.
		Raises:
			ValueError: if frame_skip or screen_size are not strictly positive.
		"""
		self.config = json_serializable(locals())
		if frame_skip <= 0:
			raise ValueError('Frame skip should be strictly positive, got {}'.
							format(frame_skip))
		if screen_size <= 0:
			raise ValueError('Target screen size should be strictly positive, got {}'.
							format(screen_size))

		self.environment = environment
		self.terminal_on_life_loss = terminal_on_life_loss
		self.frame_skip = frame_skip
		self.screen_size = screen_size

		obs_dims = self.environment.observation_space
		# Stores temporary observations used for pooling over two successive
		# frames.
		self.screen_buffer = [
			np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
			np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
		]

		self.game_over = False
		self.lives = 0  # Will need to be set by reset().

	@property
	def observation_space(self):
		# Return the observation space adjusted to match the shape of the processed
		# observations.
		return Box(low=0, high=255, shape=(self.screen_size, self.screen_size, 1),
				   dtype=np.uint8)
	@property
	def spec(self):
		return self.environment.spec

	@property
	def action_space(self):
		return self.environment.action_space

	@property
	def reward_range(self):
		return self.environment.reward_range

	@property
	def metadata(self):
		return self.environment.metadata

	def close(self):
		return self.environment.close()

	def reset(self):
		"""Resets the environment.
		Returns:
			observation: numpy array, the initial observation emitted by the
			environment.
		"""
		self.environment.reset()
		self.lives = self.environment.ale.lives()
		self._fetch_grayscale_observation(self.screen_buffer[0])
		self.screen_buffer[1].fill(0)
		return self._pool_and_resize()

	def render(self, mode):
		"""Renders the current screen, before preprocessing.
		This calls the Gym API's render() method.
		Args:
			mode: Mode argument for the environment's render() method.
				Valid values (str) are:
				'rgb_array': returns the raw ALE image.
				'human': renders to display via the Gym renderer.
		Returns:
			if mode='rgb_array': numpy array, the most recent screen.
			if mode='human': bool, whether the rendering was successful.
		"""
		return self.environment.render(mode)

	def step(self, action):
		"""Applies the given action in the environment.
		Remarks:
		* If a terminal state (from life loss or episode end) is reached, this may
			execute fewer than self.frame_skip steps in the environment.
		* Furthermore, in this case the returned observation may not contain valid
			image data and should be ignored.
		Args:
			action: The action to be executed.
		Returns:
			observation: numpy array, the observation following the action.
			reward: float, the reward following the action.
			is_terminal: bool, whether the environment has reached a terminal state.
				This is true when a life is lost and terminal_on_life_loss, or when the
				episode is over.
			info: Gym API's info data structure.
		"""
		accumulated_reward = 0.

		for time_step in range(self.frame_skip):
			# We bypass the Gym observation altogether and directly fetch the
			# grayscale image from the ALE. This is a little faster.
			_, reward, game_over, info = self.environment.step(action)
			accumulated_reward += reward

			# If we lose one life, reward - 1
			new_lives = self.environment.ale.lives()
			die = new_lives < self.lives
			if die:
				accumulated_reward -= 1
			self.lives = new_lives

			if self.terminal_on_life_loss:
				is_terminal = game_over or die
			else:
				is_terminal = game_over

			if is_terminal:
				break
			# We max-pool over the last two frames, in grayscale.
			elif time_step >= self.frame_skip - 2:
				t = time_step - (self.frame_skip - 2)
				self._fetch_grayscale_observation(self.screen_buffer[t])

		# Pool the last two observations.
		observation = self._pool_and_resize()

		self.game_over = game_over
		return observation, accumulated_reward, is_terminal, info

	def _fetch_grayscale_observation(self, output):
		"""Returns the current observation in grayscale.
		The returned observation is stored in 'output'.
		Args:
			output: numpy array, screen buffer to hold the returned observation.
		Returns:
			observation: numpy array, the current observation in grayscale.
		"""
		self.environment.ale.getScreenGrayscale(output)
		return output

	def _pool_and_resize(self):
		"""Transforms two frames into a Nature DQN observation.
		For efficiency, the transformation is done in-place in self.screen_buffer.
		Returns:
			transformed_screen: numpy array, pooled, resized screen.
		"""
		# Pool if there are enough screens to do so.
		if self.frame_skip > 1:
			np.maximum(self.screen_buffer[0], self.screen_buffer[1],
					   out=self.screen_buffer[0])

		transformed_image = cv2.resize(self.screen_buffer[0],
									   (self.screen_size, self.screen_size),
									   interpolation=cv2.INTER_AREA)
		int_image = np.asarray(transformed_image, dtype=np.uint8)
		return int_image
