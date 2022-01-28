import numpy as np
import os
import random
import tensorflow as tf
import itertools

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
		self.rewards = np.empty(self.memory_size, dtype = np.int32)
		self.screens = np.empty((self.memory_size, 84, 84), dtype = np.uint8)
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
		# NB! screen is post-state, after action and reward
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.screens[self.current, ...] = screen
		self.terminals[self.current] = terminal
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.memory_size

	def sample(self):
		assert self.count > self.history_length
		indexes = []
		while len(indexes) < self.batch_size:
			while True:
				index = random.randint(self.history_length, self.count - 1)
				# if wraps over current pointer, then get new one
				if index >= self.current and index - self.history_length < self.current:
					continue
				# if wraps over episode end, then get new one
				if self.terminals[(index - self.history_length):index].any():
					continue
				# otherwise use this index
				break
			# NB! having index first is fastest in C-order matrices
			self.states[len(indexes), ...] = self.screens[(index - self.history_length):index, ...]
			self.next_states[len(indexes), ...] = self.screens[(index - (self.history_length - 1)):(index + 1), ...]
			indexes.append(index)
		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]

		return self.states, actions, rewards, self.next_states, terminals

class DdqnValue():
	"""
		Processes a raw Atari images. Resizes it and converts it to grayscale.
	"""
	def __init__(self): 
		# Bulid the Tensorflow Graph
		with tf.variable_scope("doubledqnerror"):
			self.q_values = tf.placeholder(shape=[32, 4], dtype=tf.float32)
			self.t_values = tf.placeholder(shape=[32, 4], dtype=tf.float32)
			gather_indices = tf.range(32) * 4 + tf.argmax(self.q_values, axis=1, output_type=tf.int32)
			self.output = tf.gather(tf.reshape(self.t_values, [-1]), gather_indices)

	def process(self, sess, q_values, t_values):
		"""
		Args:
		    sess: A Tensorflow session object
		    state: A [210, 160, 3] Atari RGB State

		Returns:
		    A processed [84, 84, 1] state representing grayscale values.
		"""
		return sess.run(self.output, { self.q_values : q_values, self.t_values : t_values })

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions

class StateProcessor():
	"""
	Processes a raw Atari images. Resizes it and converts it to grayscale.
	"""
	def __init__(self): 
		with tf.variable_scope("state_processor"):
			self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
			self.output = tf.image.rgb_to_grayscale(self.input_state)
			self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
			self.output = tf.image.resize_images(
			 		self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)

	def process(self, sess, state):
		"""
		Args:
		    sess: A Tensorflow session object
		    state: A [210, 160, 3] Atari RGB State
		Returns:
		    A processed [84, 84, 1] state representing grayscale values.
		"""
		return sess.run(self.output, { self.input_state : state })

def state_process(sess, state_processor, state):
	"""
	state process, let [210, 160, 3] to [84, 84, 4]
	Args:
		sess : Tensorflow session
		state_processor : a class
		state: a rgb state
	Returns:
		 A processed [84, 84, 4] state representing grayscale values.
	"""
	state = state_processor.process(sess, state)
	return state

def copy_model_parameters(sess, estimator1, estimator2):
	"""
	Copies the model parameters of one estimator to another.

	Args:
	  sess: Tensorflow session instance
	  estimator1: Estimator to copy the paramters from
	  estimator2: Estimator to copy the parameters to
	"""
	e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
	e1_params = sorted(e1_params, key=lambda v: v.name)
	e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
	e2_params = sorted(e2_params, key=lambda v: v.name)

	update_ops = []
	for e1_v, e2_v in zip(e1_params, e2_params):
		op = e2_v.assign(e1_v)
		update_ops.append(op)

	sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
	"""
	Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

	Args:
	    estimator: An estimator that returns q values for a given state
	    nA: Number of actions in the environment.

	Returns:
	    A function that takes the (sess, observation, epsilon) as an argument and returns
	    the probabilities for each action in the form of a numpy array of length nA.

	"""
	def policy_fn(sess, observation, epsilon):
		A = np.ones(nA, dtype=float) * epsilon / nA
		q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
		best_action = np.argmax(q_values)
		A[best_action] += (1.0 - epsilon)
		return A
	return policy_fn

def populate_replay_buffer(sess, env, state_processor, replay_memory_init_size, VALID_ACTIONS, Transition, policy):
	"""
	populate replay buffer at first

	Args:
		sess:
		env:
		state_processor:
		eplay_memory_init_size:
		VALID_ACTIONS:
		Transition:
		policy:

	Return:
		replay_memory : has 'eplay_memory_init_size' steps experience.
	"""
	replay_memory = []
	history = History()
	state = env.reset()
	state_proc = state_process(sess, state_processor, state)
	for _ in range(4):
		history.add(state_proc)

	for i in range(replay_memory_init_size):
		action = np.random.choice(len(VALID_ACTIONS), p=policy(sess, history.get(), 1))
		next_state, reward, done, _ = env.step(action)
		next_state_proc = state_process(sess, state_processor, next_state)
		history.add(next_state_proc)
		experience = Transition(action, next_state_proc, reward, done)
		replay_memory.append(experience)

		if done:
			state = env.reset()
			state_proc = state_process(sess, state_processor, state)
			for _ in range(4):
				history.add(state_proc)

	return replay_memory