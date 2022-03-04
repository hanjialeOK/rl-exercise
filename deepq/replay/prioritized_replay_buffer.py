import numpy as np
import random
from collections import namedtuple, OrderedDict
import tensorflow as tf

from deepq.replay.circular_replay_buffer import ReplayMemory, ReplayElement, WrappedReplayBuffer
from deepq.replay.sum_tree import SumTree
from deepq.replay.binary_heap import *

PERDataType = namedtuple('per_data',
                         ['states', 'actions', 'rewards', 'next_states', 'terminals', 'indices', 'priorities'])


class ProportionalReplayMemory(ReplayMemory):
    def __init__(self, replay_capacity, batch_size):
        super(ProportionalReplayMemory, self).__init__(
            replay_capacity, batch_size)
        self._index_dtype = np.int32
        self._priority_dtype = np.float32
        self._sum_tree = SumTree(replay_capacity)

        # transition_elements
        self.transition_elements.append(
            ReplayElement('indices', (self._batch_size,), self._index_dtype))
        self.transition_elements.append(
            ReplayElement('priorities', (self._batch_size,), self._priority_dtype))

    def add(self, action, screen, reward, terminal, priority):
        # Note: sum_tree.set should be executed before super()
        # because self._current will +1 in super()
        self._sum_tree.set(self._current, priority)
        super(ProportionalReplayMemory, self).add(
            action, screen, reward, terminal)

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
        assert self.count > self._history_length
        indices = self._sum_tree.stratified_sample(self._batch_size)
        attempt_count = 0
        for i in range(len(indices)):
            index = indices[i]
            if not self._is_valid_index(index):
                while True:
                    index = self._sum_tree.sample()
                    if self._is_valid_index(index):
                        break
                    attempt_count += 1
                    if attempt_count >= max_sample_attempts:
                        raise RuntimeError(
                            'Max sample attempts: Tried {} times but only sampled {}'
                            ' valid indices. Batch size is {}'.
                            format(max_sample_attempts, i, self._batch_size))
                indices[i] = index
            self._states[i, ...] = \
                self._screens[(index - self._history_length):index, ...]
            self._next_states[i, ...] = \
                self._screens[(index - (self._history_length - 1))
                               :(index + 1), ...]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        terminals = self._terminals[indices]
        indices = np.asarray(indices, dtype=self._index_dtype)
        priorities = self.get_priority(indices)

        return PERDataType(self._states, actions, rewards, self._next_states, terminals,
                           indices, priorities)

    def max_priority(self):
        return self._sum_tree.max_recorded_priority

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al.
        Args:
            indices: np.array with dtype int32, of indices in range
                [0, replay_capacity).
            priorities: float, the corresponding priorities.
        """
        assert indices.dtype == np.int32, \
            ('indices must be int32s, given: {}'.format(indices.dtype))
        assert priorities.dtype == np.float32, \
            ('indices must be float32s, given: {}'.format(priorities.dtype))
        for index, priority in zip(indices, priorities):
            self._sum_tree.set(index, priority)

    def get_priority(self, indices):
        assert indices.dtype == np.int32, \
            ('indices must be int32s, given: {}'.format(indices.dtype))
        priority_batch = np.empty(
            (self._batch_size), dtype=self._priority_dtype)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self._sum_tree.get(memory_index)
        return priority_batch


class WrappedProportionalReplayBuffer(WrappedReplayBuffer):
    """Wrapper of ReplayBuffer with an in-graph sampling mechanism.
    Usage:
        To add a transition: Call the add function.
        To sample a batch: Construct operations that depend on any of the
                           tensors is the transition dictionary. Every sess.run
                           that requires any of these tensors will sample a new
                           transition.
    """

    def __init__(self, replay_capacity, batch_size):
        super(WrappedProportionalReplayBuffer, self).__init__(
            replay_capacity=replay_capacity,
            batch_size=batch_size)

    def _build_memory(self, replay_capacity, batch_size):
        return ProportionalReplayMemory(replay_capacity=replay_capacity,
                                        batch_size=batch_size)

    def add(self, action, screen, reward, terminal, priority):
        self.memory.add(action, screen, reward, terminal, priority)

    def tf_set_priority(self, indices, priorities):
        return tf.numpy_function(
            func=self.memory.set_priority, inp=[indices, priorities], Tout=[],
            name='prioritized_replay_set_priority_py_func')

    # def get_priority(self, indices):
    #     return tf.numpy_function(
    #         func=self.memory.get_priority, inp=[indices],
    #         Tout=tf.float32,
    #         name='prioritized_replay_get_priority_py_func')
