import numpy as np
import random
from collections import namedtuple

from lib.replay.sum_tree import SumTree
from lib.replay.binary_heap import *

SampleDataType = namedtuple('sample_data', \
	['states', 'actions', 'rewards', 'next_states', 'terminals'])

PERDataType = namedtuple('per_data', \
	['states', 'actions', 'rewards', 'next_states', 'terminals', 'indexes', 'prioritizes'])

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

	def _is_valid_index(self, index):
		if index < self.history_length or index >= self.count:
			return False
		if index >= self.current and index - self.history_length < self.current:
			return False
		# if wraps over episode end, then get new one
		if self.terminals[(index - self.history_length):index].any():
			return False
		return True

	def sample(self, max_sample_attempts=1000):
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
		Note: 
			if t5 is True, s5 will be a bad observation. However, 
		    target = r5 + gamma * (1 - t5) * q_max(s5) = r5, which has no business with s5.
		"""
		assert self.count > self.history_length
		indexes = []
		attempt_count = 0
		while len(indexes) < self.batch_size:
			while True:
				index = random.randint(self.history_length, self.count - 1)
				if self._is_valid_index(index):
					break
				attempt_count += 1
				if attempt_count >= max_sample_attempts:
					raise RuntimeError(
						'Max sample attempts: Tried {} times but only sampled {}'
						' valid indices. Batch size is {}'.
						format(max_sample_attempts, len(indexes), self.batch_size))
			# NB! having index first is fastest in C-order matrices
			self.states[len(indexes), ...] = \
				self.screens[(index - self.history_length):index, ...]
			self.next_states[len(indexes), ...] = \
				self.screens[(index - (self.history_length - 1)):(index + 1), ...]
			indexes.append(index)
		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]

		return SampleDataType(self.states, actions, rewards, self.next_states, terminals)

class ProportionalReplay(ReplayMemory):
	def __init__(self, memory_size, batch_size):
		super(ProportionalReplay, self).__init__(memory_size, batch_size)
		self.sum_tree = SumTree(memory_size)

	def add(self, action, screen, reward, terminal, priority):
		# Note: sum_tree.set should be executed before super()
		# because self.current will +1 in super()
		self.sum_tree.set(self.current, priority)
		super(ProportionalReplay, self).add(action, screen, reward, terminal)

	def sample(self, max_sample_attempts=1000):
		"""
		This process looks like:
			...
			<a0, s1, r1, t1, p1>
			<a1, s2, r2, t2, p2>
			<a2, s3, r3, t3, p3>
			<a3, s4, r4, t4, p4>
			<a4, s5, r5, t5, p5>  <=  index
			...
		Returns:
			state = [s1, s2, s3, s4]
			next_state = [s2, s3, s4, s5]
			action = a4
			reward = r5
			terminal = t5
			priority = p5
		Note: 
			if t5 is True, s5 will be a bad observation. However, 
		    target = r5 + gamma * (1 - t5) * q_max(s5) = r5, which has no business with s5.
		"""
		assert self.count > self.history_length
		indexes = self.sum_tree.stratified_sample(self.batch_size)
		attempt_count = 0
		for i in range(len(indexes)):
			index = indexes[i]
			if not self._is_valid_index(index):
				while True:
					index = self.sum_tree.sample()
					if self._is_valid_index(index):
						break
					attempt_count += 1
					if attempt_count >= max_sample_attempts:
						raise RuntimeError(
							'Max sample attempts: Tried {} times but only sampled {}'
							' valid indices. Batch size is {}'.
							format(max_sample_attempts, i, self.batch_size))
				indexes[i] = index
			self.states[i, ...] = \
				self.screens[(index - self.history_length):index, ...]
			self.next_states[i, ...] = \
				self.screens[(index - (self.history_length - 1)):(index + 1), ...]
		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]
		indexes = np.asarray(indexes, dtype=np.int32)
		priorities = self.get_priority(indexes)

		return PERDataType(self.states, actions, rewards, self.next_states, terminals, \
						   indexes, priorities)

	def max_priority(self):
		return self.sum_tree.max_recorded_priority

	def set_priority(self, indexes, priorities):
		"""Sets the priority of the given elements according to Schaul et al.
		Args:
			indexes: np.array with dtype int32, of indexes in range
				[0, replay_capacity).
			priorities: float, the corresponding priorities.
		"""
		assert indexes.dtype == np.int32, \
			('indexes must be int32s, given: {}'.format(indexes.dtype))
		assert priorities.dtype == np.float32, \
			('indexes must be float32s, given: {}'.format(priorities.dtype))
		for index, priority in zip(indexes, priorities):
			self.sum_tree.set(index, priority)

	def get_priority(self, indexes):
		assert indexes.dtype == np.int32, \
			('indexes must be int32s, given: {}'.format(indexes.dtype))
		priority_batch = np.empty((self.batch_size), dtype=np.float32)
		for i, memory_index in enumerate(indexes):
			priority_batch[i] = self.sum_tree.get(memory_index)
		return priority_batch

class RankBasedReplay(ReplayMemory):
	def __init__(self, memory_size, batch_size):
		super(RankBasedReplay, self).__init__(memory_size, batch_size)
		self.heap = ArrayBasedHeap(memory_size)

	def add(self, action, screen, reward, terminal, priority):
		self.heap.insert(HeapItem(priority, self.current))
		super(ProportionalReplay, self).add(action, screen, reward, terminal)

	def sample(self, max_sample_attempts=1000):
		"""
		This process looks like:
			...
			<a0, s1, r1, t1, p1>
			<a1, s2, r2, t2, p2>
			<a2, s3, r3, t3, p3>
			<a3, s4, r4, t4, p4>
			<a4, s5, r5, t5, p5>  <=  index
			...
		Returns:
			state = [s1, s2, s3, s4]
			next_state = [s2, s3, s4, s5]
			action = a4
			reward = r5
			terminal = t5
			priority = p5
		Note: 
			if t5 is True, s5 will be a bad observation. However, 
		    target = r5 + gamma * (1 - t5) * q_max(s5) = r5, which has no business with s5.
		"""
		assert self.count > self.history_length
		indexes, priorities = self.heap.stratified_sample(self.batch_size)
		attempt_count = 0
		for i in range(len(indexes)):
			index = indexes[i]
			if not self._is_valid_index(index):
				while True:
					index, priority = self.heap.sample()
					if self._is_valid_index(index):
						break
					attempt_count += 1
					if attempt_count >= max_sample_attempts:
						raise RuntimeError(
							'Max sample attempts: Tried {} times but only sampled {}'
							' valid indices. Batch size is {}'.
							format(max_sample_attempts, i, self.batch_size))
				indexes[i] = index
				priorities[i] = priority
			self.states[i, ...] = \
				self.screens[(index - self.history_length):index, ...]
			self.next_states[i, ...] = \
				self.screens[(index - (self.history_length - 1)):(index + 1), ...]
		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]
		indexes = np.asarray(indexes, dtype=np.int32)
		priorities = np.asarray(priorities, dtype=np.float32)

		return PERDataType(self.states, actions, rewards, self.next_states, terminals, \
						   indexes, priorities)

	def set_priority(self, indexes, priorities):
		pass