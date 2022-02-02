import numpy as np
import random
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
