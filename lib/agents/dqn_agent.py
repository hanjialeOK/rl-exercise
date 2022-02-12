import numpy as np
import os
import random
import tensorflow as tf

from lib.utils import json_serializable, History
from lib.replay.circular_replay_buffer import WrappedReplayBuffer
from lib.replay.prioritized_replay_buffer import WrappedProportionalReplayBuffer

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
        Begin at 1. until warmup_steps steps have been taken; then
        Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
        Use epsilon from there on.
    Args:
        decay_period: float, the period over which epsilon is decayed.
        step: int, the number of training steps completed so far.
        warmup_steps: int, the number of steps taken before epsilon is decayed.
        epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
        A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus

class WrappedRMSPropOptimizer(tf.train.RMSPropOptimizer):
    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 centered=False,
                 name="RMSProp"):
        self.config = json_serializable(locals())
        super(WrappedRMSPropOptimizer, self).__init__(
            learning_rate=learning_rate,
            decay=decay,
            momentum=momentum,
            epsilon=epsilon,
            centered=centered,
            name=name)

    def get_config(self):
        return self.config

class WrappedAdamOptimizer(tf.train.AdamOptimizer):
    def __init__(self,
                 learning_rate=0.001,
                 epsilon=1e-08,
                 name='Adam'):
        self.config = json_serializable(locals())
        super(WrappedAdamOptimizer, self).__init__(
            learning_rate=learning_rate,
            epsilon=epsilon,
            name=name)

    def get_config(self):
        return self.config

class NatureDQNNetwork(tf.keras.Model):
    """The convolutional network used to compute the agent's Q-values."""

    def __init__(self, num_actions, name=None):
        """
        Creates the layers used for calculating Q-values.
        Args:
            num_actions: int, number of actions.
            name: str, used to create scope for network parameters.
        """
        super(NatureDQNNetwork, self).__init__(name=name)

        self._num_actions = num_actions
        # Defining layers.
        activation_fn = tf.keras.activations.relu
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
              data_format='channels_first', activation=activation_fn, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
              data_format='channels_first', activation=activation_fn, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
              data_format='channels_first', activation=activation_fn, name='conv3')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                            name='fc1')
        self.dense2 = tf.keras.layers.Dense(self._num_actions, name='fc2')

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
        x = self.dense1(x)
        x = self.dense2(x)
        return x

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
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
              data_format='channels_first', activation=activation_fn, name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
              data_format='channels_first', activation=activation_fn, name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
              data_format='channels_first', activation=activation_fn, name='conv3')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                            name='fc1')
        self.dense2 = tf.keras.layers.Dense(self._num_actions, name='fc2')
        self.dense3 = tf.keras.layers.Dense(512, activation=activation_fn,
                                            name='fc3')
        self.dense4 = tf.keras.layers.Dense(1, name='fc4')

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
        x = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
        return x

class DQNAgent():
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
                 optimizer=WrappedRMSPropOptimizer(
                     learning_rate=0.00025, \
                     decay=0.95, \
                     epsilon=1e-6, \
                     centered=False),
                 summary_writer=None,
                 summary_writing_frequency=500):
        self.config = json_serializable(locals())
        self.config['agent_type'] = self.__class__.__name__
        self.config['optimizer'] = optimizer.get_config()
        self._sess = sess
        self._num_actions = num_actions
        self._replay_capacity = replay_capacity
        self._replay_min_size = replay_min_size
        self._update_period = update_period
        self._target_update_period = target_update_period
        self._epsilon_train = epsilon_train
        self._epsilon_eval = epsilon_eval
        self._epsilon_decay_period = epsilon_decay_period
        self._gamma = gamma
        self._batch_size = batch_size
        self._eval_mode = eval_mode
        self._max_tf_checkpoints_to_keep = max_tf_checkpoints_to_keep
        self._optimizer = optimizer
        self._summary_writer = summary_writer
        self._summary_writing_frequency = summary_writing_frequency
        self._history = History()
        self._replay = self._build_replay_buffer()
        self._build_networks()
        self._train_op = self._build_train_op()
        self._sync_qt_ops = self._build_sync_op()
        self._state_processer = self._build_state_processer()
        self._saver = self._build_saver()

        if self._summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self._merged_summaries = tf.summary.merge_all()

        self._training_steps = 0

    def _build_replay_buffer(self):
        return WrappedReplayBuffer(replay_capacity=self._replay_capacity, \
                                   batch_size=self._batch_size)

    def _build_networks(self):
        self.online_network = NatureDQNNetwork(num_actions=self._num_actions, name="online")
        self.target_network = NatureDQNNetwork(num_actions=self._num_actions, name="target")

    # Note: Required to be called after _build_train_op(), otherwise return []
    def _get_var_list(self, name='online'):
        scope = tf.get_default_graph().get_name_scope()
        trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, name))
        return trainable_variables

    def _build_target_q_op(self):
        target_next_q_max = tf.reduce_max(self.target_net_replay_output, axis=1) # (32,)
        target = self.replay_rewards + \
                 self._gamma * (1.0 - self.replay_terminals) * target_next_q_max # (32,)
        return target

    def _build_train_op(self):
        self.state_ph = tf.placeholder(shape=[1, 4, 84, 84], dtype=tf.uint8, name="state_ph")
        self.replay_states = tf.cast(self._replay.transition['states'], tf.uint8)
        self.replay_actions = tf.cast(self._replay.transition['actions'], tf.int32)
        self.replay_rewards = tf.cast(self._replay.transition['rewards'], tf.float32)
        self.replay_next_states = tf.cast(self._replay.transition['next_states'], tf.uint8)
        self.replay_terminals = tf.cast(self._replay.transition['terminals'], tf.float32)

        self.online_net_output = self.online_network(self.state_ph) # (1, 4)
        self.online_net_replay_output = self.online_network(self.replay_states) # (32, 4)
        self.target_net_replay_output = self.target_network(self.replay_next_states) # (32, 4)
        self.q_argmax = tf.argmax(self.online_net_output, axis=1)[0] # (1, ) => ()

        # Target
        target_nograd = tf.stop_gradient(self._build_target_q_op())

        # Online
        q_value_chosen_2d = tf.gather(self.online_net_replay_output, \
                                       tf.expand_dims(self.replay_actions, axis=-1), \
                                      axis=1, batch_dims=1) # (32, 1)
        q_value_chosen = tf.squeeze(q_value_chosen_2d) # (32,)

        losses = tf.losses.huber_loss(
            target_nograd, q_value_chosen, reduction=tf.losses.Reduction.NONE)
        loss = tf.reduce_mean(losses)
        if self._summary_writer is not None:
            with tf.variable_scope('losses'):
                tf.summary.scalar(name='huberloss', tensor=loss)
            with tf.variable_scope('q_estimate'):
                tf.summary.scalar(name='max_q_value', \
                                  tensor=tf.reduce_max(self.online_net_replay_output))
                tf.summary.scalar(name='avg_q_value', \
                                  tensor=tf.reduce_mean(self.online_net_replay_output))
        return self._optimizer.minimize(loss, var_list=self._get_var_list())

    # These are legacy and should probably be removed in future versions.
    def _build_state_processer(self):
        self.raw_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
        self.output = tf.image.rgb_to_grayscale(self.raw_state)
        self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
        self.output = tf.image.resize(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.squeeze(self.output)

    def _build_sync_op(self):
        """Builds ops for assigning weights from online to target network.
        Returns:
            ops: A list of ops assigning weights from online to target network.
        """
        # Get trainable variables from online and target DQNs
        sync_qt_ops = []
        trainable_online = self._get_var_list()
        trainable_target = self._get_var_list('target')

        for (w_online, w_target) in zip(trainable_online, trainable_target):
            sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
        return sync_qt_ops

    def _build_saver(self):
        return tf.train.Saver(var_list=self._get_var_list(), \
                max_to_keep=self._max_tf_checkpoints_to_keep)

    def bundle(self, checkpoint_dir, iteration):
        if not os.path.exists(checkpoint_dir):
            raise
        self.online_network.save_weights(
            os.path.join(checkpoint_dir, 'best_model.h5'), save_format='h5')
        self._saver.save(
            self._sess, \
            os.path.join(checkpoint_dir, 'tf_ckpt'), \
            global_step=iteration)

    def unbundle(self, checkpoint_dir, iteration=None):
        if not os.path.exists(checkpoint_dir):
            raise
        # Load the best weights without iteraion.
        if iteration is None:
            self.online_network.load_weights(
                os.path.join(checkpoint_dir, 'best_model.h5'))
        else:
            self._saver.restore(
                self._sess, \
                os.path.join(checkpoint_dir, f'tf_ckpt-{iteration}'))

    def select_action(self):
        if self._eval_mode:
            epsilon = self._epsilon_eval
        else:
            epsilon = linearly_decaying_epsilon(
                decay_period=self._epsilon_decay_period,
                step=self._training_steps,
                warmup_steps=self._replay_min_size,
                epsilon=self._epsilon_train)
        if random.random() <= epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return self._sess.run(self.q_argmax, {self.state_ph: self._history.get()})

    def _store_transition(self, action, observation, reward, terminal):
        self._replay.add(action, observation, reward, terminal)

    def step(self, action, observation, reward, terminal):
        self._history.add(observation)
        # If eval, store and train are no longer needed.
        if self._eval_mode:
            return
        # Store transition
        self._store_transition(action, observation, reward, terminal)
        # Train
        if self._replay.memory.count > self._replay_min_size:
            if self._training_steps % self._update_period == 0:
                self._sess.run(self._train_op)

                if self._summary_writer is not None and \
                    self._training_steps % self._summary_writing_frequency == 0:
                    summary = self._sess.run(self._merged_summaries)
                    self._summary_writer.add_summary(summary, self._training_steps)

            if self._training_steps % self._target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

        self._training_steps += 1

    def begin_episode(self, observation):
        for _ in range(4):
            self._history.add(observation)

class DDQNAgent(DQNAgent):
    def __init__(self, sess, num_actions, summary_writer):
        super(DDQNAgent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            summary_writer=summary_writer)

    def _build_target_q_op(self):
        online_next_q = self.online_network(self.replay_next_states) # (32, 4)
        next_q_2d = tf.gather(self.target_net_replay_output, \
                              tf.expand_dims(tf.argmax(online_next_q, axis=1), axis=-1), \
                              axis=1, batch_dims=1) # (32, 1)
        next_q = tf.squeeze(next_q_2d) # (32, )
        target = self.replay_rewards + self._gamma * (1 - self.replay_terminals) * next_q # (32,)
        return target

class PERAgent(DQNAgent):
    def __init__(self, sess, num_actions, summary_writer, replay_scheme='uniform'):
        self._replay_scheme = replay_scheme
        super(PERAgent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            optimizer=WrappedAdamOptimizer(
                learning_rate=0.00025, epsilon=0.0003125),
            summary_writer=summary_writer)

    def _build_replay_buffer(self):
        return WrappedProportionalReplayBuffer(replay_capacity=self._replay_capacity, \
                                               batch_size=self._batch_size)

    def _build_train_op(self):
        self.state_ph = tf.placeholder(shape=[1, 4, 84, 84], dtype=tf.uint8, name="state_ph")
        self.replay_states = tf.cast(self._replay.transition['states'], tf.uint8)
        self.replay_actions = tf.cast(self._replay.transition['actions'], tf.int32)
        self.replay_rewards = tf.cast(self._replay.transition['rewards'], tf.float32)
        self.replay_next_states = tf.cast(self._replay.transition['next_states'], tf.uint8)
        self.replay_terminals = tf.cast(self._replay.transition['terminals'], tf.float32)
        self.replay_indices = self._replay.transition['indices']
        self.replay_probs = tf.cast(self._replay.transition['priorities'], tf.float32)

        self.online_net_output = self.online_network(self.state_ph) # (1, 4)
        self.online_net_replay_output = self.online_network(self.replay_states) # (32, 4)
        self.target_net_replay_output = self.target_network(self.replay_next_states) # (32, 4)
        self.q_argmax = tf.argmax(self.online_net_output, axis=1)[0] # (1, ) => ()

        # Target
        target_nograd = tf.stop_gradient(self._build_target_q_op())

        # Online
        q_value_chosen_2d = tf.gather(self.online_net_replay_output, \
                                       tf.expand_dims(self.replay_actions, axis=-1), \
                                      axis=1, batch_dims=1) # (32, 1)
        q_value_chosen = tf.squeeze(q_value_chosen_2d) # (32,)

        losses = tf.losses.huber_loss(
            target_nograd, q_value_chosen, reduction=tf.losses.Reduction.NONE)

        if self._replay_scheme == 'prioritized':
            # The original prioritized experience replay uses a linear exponent
            # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
            # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
            # a fixed exponent actually performs better, except on Pong.
            loss_weights = 1.0 / tf.sqrt(self.replay_probs + 1e-10)
            loss_weights /= tf.reduce_max(loss_weights)

            # Rainbow and prioritized replay are parametrized by an exponent alpha,
            # but in both cases it is set to 0.5 - for simplicity's sake we leave it
            # as is here, using the more direct tf.sqrt(). Taking the square root
            # "makes sense", as we are dealing with a squared loss.
            # Add a small nonzero value to the loss to avoid 0 priority items. While
            # technically this may be okay, setting all items to 0 priority will cause
            # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
            update_priorities_op = self._replay.tf_set_priority(
                self.replay_indices, tf.sqrt(losses + 1e-10))

            # Weight the loss by the inverse priorities.
            losses = loss_weights * losses
        else:
            update_priorities_op = tf.no_op()

        with tf.control_dependencies([update_priorities_op]):
            loss = tf.reduce_mean(losses)
            if self._summary_writer is not None:
                with tf.variable_scope('losses'):
                    tf.summary.scalar(name='huberloss', tensor=loss)
                with tf.variable_scope('q_estimate'):
                    tf.summary.scalar(name='max_q_value', \
                                    tensor=tf.reduce_max(self.online_net_replay_output))
                    tf.summary.scalar(name='avg_q_value', \
                                    tensor=tf.reduce_mean(self.online_net_replay_output))
            return self._optimizer.minimize(loss, var_list=self._get_var_list())

    def _store_transition(self, action, observation, reward, terminal):
        if self._replay_scheme == 'uniform':
            priority = 1.
        else:
            priority = self._replay.memory.max_priority()
        self._replay.add(action, observation, reward, terminal, priority)

class DuelingAgent(DQNAgent):
    def __init__(self, sess, num_actions, summary_writer):
        super(DuelingAgent, self).__init__(
            sess=sess,
            num_actions=num_actions,
            summary_writer=summary_writer)

    def _build_networks(self):
        self.online_network = DuelingNetwork(num_actions=self._num_actions, name="online")
        self.target_network = DuelingNetwork(num_actions=self._num_actions, name="target")