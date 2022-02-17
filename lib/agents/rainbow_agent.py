import numpy as np
import os
import random
import tensorflow as tf
from collections import namedtuple

from lib.agents.dqn_agent import DQNAgent, WrappedAdamOptimizer

C51NetworkType = namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])


class C51Network(tf.keras.Model):
    """The convolutional network used to compute agent's return distributions."""

    def __init__(self, num_actions, num_atoms, support, name=None):
        """Creates the layers used calculating return distributions.
        Args:
            num_actions: int, number of actions.
            num_atoms: int, the number of buckets of the value function distribution.
            support: tf.linspace, the support of the Q-value distribution.
            name: str, used to crete scope for network parameters.
        """
        super(C51Network, self).__init__(name=name)
        activation_fn = tf.keras.activations.relu
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.support = support
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
        # Defining layers.
        self.conv1 = tf.keras.layers.Conv2D(
            32, [8, 8], strides=4, padding='same',
            data_format='channels_first', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            64, [4, 4], strides=2, padding='same',
            data_format='channels_first', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(
            64, [3, 3], strides=1, padding='same',
            data_format='channels_first', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='conv3')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            512, activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
            name='fc2')

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input.
        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.
        Args:
            state: Tensor, input tensor.
        Returns:
            collections.namedtuple, output ops (graph mode) or output tensors (eager).
        """
        x = tf.cast(state, tf.float32)
        x = x / 255
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
        probabilities = tf.keras.activations.softmax(logits)
        q_values = tf.reduce_sum(self.support * probabilities, axis=2)
        return C51NetworkType(q_values, logits, probabilities)


class C51Agent(DQNAgent):
    def __init__(self,
                 sess,
                 num_actions,
                 replay_capacity=1000000,
                 replay_min_size=50000,
                 update_period=4,
                 target_update_period=10000,
                 epsilon_train=0.01,
                 epsilon_eval=0.001,
                 epsilon_decay_period=1000000,
                 gamma=0.99,
                 batch_size=32,
                 eval_mode=False,
                 max_tf_checkpoints_to_keep=4,
                 optimizer=WrappedAdamOptimizer(
                     learning_rate=0.00025,
                     epsilon=0.0003125),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 num_atoms=51,
                 vmin=None,
                 vmax=10.):
        vmax = float(vmax)
        vmin = vmin if vmin else -vmax
        self._num_atoms = num_atoms
        self._support = tf.linspace(vmin, vmax, num_atoms)  # (51,)
        super(C51Agent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            replay_capacity=replay_capacity,
            replay_min_size=replay_min_size,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            gamma=gamma,
            batch_size=batch_size,
            eval_mode=eval_mode,
            max_tf_checkpoints_to_keep=max_tf_checkpoints_to_keep,
            optimizer=optimizer,
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency)

    def _create_network(self, name):
        network = C51Network(num_actions=self._num_actions,
                             num_atoms=self._num_atoms,
                             support=self._support,
                             name=name)
        return network

    def _build_target_distribution(self):
        """Builds the C51 target distribution as per Bellemare et al. (2017).
        First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
        is the support of the next state distribution:
        * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
        * 0 otherwise (duplicated num_atoms times).
        Second, we compute the next-state probabilities, corresponding to the action
        with highest expected value.
        Finally we project the Bellman target (support + probabilities) onto the
        original support.
        Returns:
            target_distribution: tf.tensor, the target distribution from the replay.
        """
        batch_size = self._batch_size

        # size of tiled_support: batch_size x num_atoms
        tiled_support = tf.tile(self._support, [batch_size])  # (51x32,)
        tiled_support = tf.reshape(
            tiled_support, [batch_size, self._num_atoms])  # (32, 51)

        target_support = self.replay_rewards[:, None] + \
            self._gamma * \
            (1.0 - self.replay_terminals[:, None]) * tiled_support  # (32, 51)

        next_qt_argmax = tf.argmax(
            self.target_net_replay_output.q_values, axis=1)[:, None]  # (32, 1)
        batch_indices = tf.range(
            tf.cast(batch_size, tf.int64))[:, None]  # (32, 1)
        batch_indexed_next_qt_argmax = tf.concat(
            [batch_indices, next_qt_argmax], axis=1)  # (32, 2)

        # next_probabilities (32, 51)
        next_probabilities = tf.gather_nd(
            self.target_net_replay_output.probabilities,  # (32, 4, 51)
            batch_indexed_next_qt_argmax)  # (32, 2)

        return self._project_distribution(target_support, next_probabilities, self._support)

    def _build_train_op(self):
        self.q_argmax = tf.argmax(
            self.online_net_output.q_values, axis=1)[0]  # ()

        # Target
        target_distribution = tf.stop_gradient(
            self._build_target_distribution())  # (32, 51)
        # Online
        indices = tf.range(
            tf.shape(self.online_net_replay_output.logits)[0])[:, None]  # (32, 1)
        reshaped_actions = tf.concat(
            [indices, self.replay_actions[:, None]], axis=1)  # (32, 2)
        chosen_action_logits = tf.gather_nd(self.online_net_replay_output.logits,
                                            reshaped_actions)  # (32, 51)

        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_distribution,
            logits=chosen_action_logits)
        loss = tf.reduce_mean(losses)
        grads_and_vars = self._optimizer.compute_gradients(
            loss, var_list=self._get_var_list())
        capped_grads_and_vars = [(tf.clip_by_norm(grad, 10.0), var)
                                 for grad, var in grads_and_vars]
        if self._summary_writer is not None:
            with tf.variable_scope('losses'):
                tf.summary.scalar(name='CrossEntropyLoss', tensor=loss)
            with tf.variable_scope('q_estimate'):
                tf.summary.scalar(name='max_q_value',
                                  tensor=tf.reduce_max(self.online_net_replay_output.q_values))
                tf.summary.scalar(name='avg_q_value',
                                  tensor=tf.reduce_mean(self.online_net_replay_output.q_values))
            with tf.variable_scope('rainbow_grads'):
                for grad, var in grads_and_vars:
                    tf.summary.histogram(name=f"{var.name}", values=grad)
            with tf.variable_scope('rainbow_grads_l2norm'):
                for grad, var in grads_and_vars:
                    tf.summary.scalar(name=f"{var.name}",
                                      tensor=tf.norm(tensor=grad, ord=2))
        return self._optimizer.apply_gradients(capped_grads_and_vars)

    def _project_distribution(self, supports, weights, target_support):
        target_support_deltas = target_support[1:] - target_support[:-1]
        delta_z = target_support_deltas[0]
        v_min, v_max = target_support[0], target_support[-1]
        batch_size = tf.shape(supports)[0]
        num_dims = tf.shape(target_support)[0]
        clipped_support = tf.clip_by_value(supports, v_min, v_max)[
            :, None, :]  # (32, 1, 51)
        tiled_support = tf.tile(
            [clipped_support], [1, 1, num_dims, 1])  # (1, 32, 51, 51)
        reshaped_target_support = tf.tile(target_support[:, None], [
                                          batch_size, 1])  # (51x32, 1)
        reshaped_target_support = tf.reshape(reshaped_target_support,
                                             [batch_size, num_dims, 1])  # (32, 51, 1)
        # (1, 32, 51, 51)
        numerator = tf.abs(tiled_support - reshaped_target_support)
        quotient = 1 - (numerator / delta_z)
        clipped_quotient = tf.clip_by_value(quotient, 0, 1)  # (1, 32, 51, 51)
        weights = weights[:, None, :]  # (32, 1, 51)
        inner_prod = clipped_quotient * weights  # (1, 32, 51, 51)
        projection = tf.reduce_sum(inner_prod, 3)  # (1, 32, 51)
        projection = tf.reshape(projection, [batch_size, num_dims])  # (32, 51)
        return projection
