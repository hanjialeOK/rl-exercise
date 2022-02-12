import numpy as np
import random
from collections import namedtuple, OrderedDict
import tensorflow as tf

ReplayDataType = namedtuple('sample_data', \
    ['states', 'actions', 'rewards', 'next_states', 'terminals'])

ReplayElement = namedtuple('shape_type', ['name', 'shape', 'type'])

class ReplayMemory:
    def __init__(self, replay_capacity, batch_size):
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._screen_shape = (84, 84)
        self._action_dtype = np.int32
        self._screen_dtype = np.uint8
        self._reward_dtype = np.float32
        self._terminal_dtype = bool
        # Note: screen is next_state, after action
        self._actions = np.empty(self._replay_capacity, dtype=self._action_dtype)
        self._screens = np.empty((self._replay_capacity,) + self._screen_shape, dtype=self._screen_dtype)
        self._rewards = np.empty(self._replay_capacity, dtype=self._reward_dtype)
        self._terminals = np.empty(self._replay_capacity, dtype=self._terminal_dtype)
        self._history_length = 4
        self._current = 0
        self.count = 0

        # pre-allocate states and next_states for minibatch
        self._states = np.empty((self._batch_size, self._history_length) + self._screen_shape, \
                                dtype=self._screen_dtype)
        self._next_states = np.empty((self._batch_size, self._history_length) + self._screen_shape, \
                                    dtype=self._screen_dtype)
        # transition_elements
        self.transition_elements = [
            ReplayElement('states', (self._batch_size, self._history_length) + self._screen_shape, \
                          self._screen_dtype),
            ReplayElement('actions', (self._batch_size,), self._action_dtype),
            ReplayElement('rewards', (self._batch_size,), self._reward_dtype),
            ReplayElement('next_states', (self._batch_size, self._history_length) + self._screen_shape, \
                          self._screen_dtype),
            ReplayElement('terminals', (self._batch_size,), self._terminal_dtype)
        ]

    def add(self, action, screen, reward, terminal):
        assert screen.shape == self._screen_shape
        # Note: screen is next_state, after action
        self._actions[self._current] = action
        self._screens[self._current, ...] = screen
        self._rewards[self._current] = reward
        self._terminals[self._current] = terminal
        # Note: count must be caculated bofore _current.
        self.count = max(self.count, self._current + 1)
        self._current = (self._current + 1) % self._replay_capacity

    def _is_valid_index(self, index):
        if index < self._history_length or index >= self.count:
            return False
        # if wraps over old and new, then get new one
        if index >= self._current and index - self._history_length < self._current:
            return False
        # if wraps over episode end, then get new one
        if self._terminals[(index - self._history_length):index].any():
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
        assert self.count > self._history_length
        indices = []
        attempt_count = 0
        while len(indices) < self._batch_size:
            while True:
                index = random.randint(self._history_length, self.count - 1)
                if self._is_valid_index(index):
                    break
                attempt_count += 1
                if attempt_count >= max_sample_attempts:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.
                        format(max_sample_attempts, len(indices), self._batch_size))
            # NB! having index first is fastest in C-order matrices
            self._states[len(indices), ...] = \
                self._screens[(index - self._history_length):index, ...]
            self._next_states[len(indices), ...] = \
                self._screens[(index - (self._history_length - 1)):(index + 1), ...]
            indices.append(index)
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        terminals = self._terminals[indices]

        return ReplayDataType(self._states, actions, rewards, self._next_states, terminals)

class WrappedReplayBuffer:
    """Wrapper of ReplayBuffer with an in-graph sampling mechanism.
    Usage:
        To add a transition: Call the add function.
        To sample a batch: Construct operations that depend on any of the
                           tensors is the transition dictionary. Every sess.run
                           that requires any of these tensors will sample a new
                           transition.
    """
    def __init__(self, replay_capacity, batch_size):
        self.memory = self._build_memory(replay_capacity, batch_size)
        self.transition = OrderedDict()
        self._create_sample_ops()

    def _build_memory(self, replay_capacity, batch_size):
        return ReplayMemory(replay_capacity=replay_capacity,
                            batch_size=batch_size)

    def add(self, action, screen, reward, terminal):
        self.memory.add(action, screen, reward, terminal)

    def _create_sample_ops(self):
        with tf.name_scope('sample_replay'):
            transition_type = self.memory.transition_elements
            transition_tensors = tf.numpy_function(
                func=self.memory.sample, inp=[],
                Tout=[return_entry.type for return_entry in transition_type],
                name='replay_sample_py_func')
            # Note: restore shapes lost due to applying a numpy_function.
            for element, element_type in zip(transition_tensors, transition_type):
                element.set_shape(element_type.shape)
                self.transition[element_type.name] = element
