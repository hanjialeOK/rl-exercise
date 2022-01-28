import numpy as np
import random

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
