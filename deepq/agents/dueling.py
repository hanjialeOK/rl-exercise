import numpy as np
import tensorflow as tf

from deepq.agents.dqn import DQNAgent


class DuelingNetwork(tf.keras.Model):
    """The convolutional network used to compute the agent's Q-values."""

    def __init__(self, num_actions, name=None):
        """
        Creates the layers used for calculating Q-values.
        Args:
            num_actions: int, number of actions.
            name: str, used to create scope for network parameters.
        """
        super(DuelingNetwork, self).__init__(name=name)

        self._num_actions = num_actions
        # Defining layers.
        activation_fn = tf.keras.activations.relu
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        self.conv1 = tf.keras.layers.Conv2D(
            32, [8, 8], strides=4, padding='same',
            data_format='channels_first', activation=activation_fn,
            kernel_initializer=kernel_initializer, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            64, [4, 4], strides=2, padding='same',
            data_format='channels_first', activation=activation_fn,
            kernel_initializer=kernel_initializer, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(
            64, [3, 3], strides=1, padding='same',
            data_format='channels_first', activation=activation_fn,
            kernel_initializer=kernel_initializer, name='conv3')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            512, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            self._num_actions, kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            512, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc3')
        self.dense4 = tf.keras.layers.Dense(
            1, kernel_initializer=kernel_initializer, name='fc4')

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input.
        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.
        Parameters created here will have scope according to the `name` argument
        given at `.__init__()` call.
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
        advantage = self.dense1(x)
        advantage = self.dense2(advantage)
        value = self.dense3(x)
        value = self.dense4(value)
        x = value + (advantage -
                     tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return x


class DuelingAgent(DQNAgent):
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
                 optimizer=tf.compat.v1.train.RMSPropOptimizer(
                     learning_rate=0.00025,
                     decay=0.95,
                     epsilon=1e-6,
                     centered=False),
                 summary_writer=None,
                 summary_writing_frequency=500):
        super(DuelingAgent, self).__init__(
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
        network = DuelingNetwork(num_actions=self._num_actions, name=name)
        return network
