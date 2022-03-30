import numpy as np
import tensorflow as tf

from deepq.agents.ddqn import DDQNAgent
from deepq.replay.prioritized_replay_buffer import WrappedProportionalReplayBuffer


def linearly_beta_schedule(schedule_timesteps, step, initial_p, final_p):
    fraction = min(float(step) / schedule_timesteps, 1.0)
    return initial_p + fraction * (final_p - initial_p)


class PERAgent(DDQNAgent):
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
                 optimizer=tf.compat.v1.train.AdamOptimizer(
                     learning_rate=0.00025/4,
                     epsilon=0.0003125),
                 summary_writer=None,
                 summary_writing_frequency=500,
                 replay_scheme='prioritized',
                 replay_alpha=0.6,
                 replay_beta=0.5):
        self._replay_scheme = replay_scheme
        self._replay_alpha = replay_alpha
        self._replay_beta = replay_beta
        super(PERAgent, self).__init__(
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

    def _build_replay_buffer(self):
        if self._replay_scheme not in ['uniform', 'prioritized']:
            raise ValueError(
                'Invalid replay scheme: {}'.format(self._replay_scheme))
        return WrappedProportionalReplayBuffer(
            replay_capacity=self._replay_capacity,
            batch_size=self._batch_size)

    def _build_train_op(self):
        self.replay_indices = self._replay.transition['indices']
        self.replay_probs = tf.cast(
            self._replay.transition['priorities'], tf.float32)

        self.q_argmax = tf.argmax(self.online_net_output, axis=1)[0]  # ()

        # Target
        target_nograd = tf.stop_gradient(self._build_target_q_op())

        # Online
        replay_actions_one_hot = tf.one_hot(
            self.replay_actions, self._num_actions, 1., 0., axis=-1)  # (32, 4)
        q_value_chosen = tf.reduce_sum(
            self.online_net_replay_output * replay_actions_one_hot, axis=1)  # (32,)

        losses = tf.losses.huber_loss(
            target_nograd, q_value_chosen, reduction=tf.losses.Reduction.NONE)

        if self._replay_scheme == 'prioritized':
            # The original prioritized experience replay uses a linear exponent
            # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
            # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
            # a fixed exponent actually performs better, except on Pong.
            loss_weights = 1.0 / (self.replay_probs ** self._replay_beta)
            loss_weights /= tf.reduce_max(loss_weights)

            # Rainbow and prioritized replay are parametrized by an exponent alpha,
            # but in both cases it is set to 0.5 - for simplicity's sake we leave it
            # as is here, using the more direct tf.sqrt(). Taking the square root
            # "makes sense", as we are dealing with a squared loss.
            # Add a small nonzero value to the loss to avoid 0 priority items. While
            # technically this may be okay, setting all items to 0 priority will cause
            # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
            update_priorities_op = self._replay.tf_set_priority(
                self.replay_indices, (losses + 1e-10) ** self._replay_alpha)

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
                    tf.summary.scalar(name='max_q_value',
                                      tensor=tf.reduce_max(self.online_net_replay_output))
                    tf.summary.scalar(name='avg_q_value',
                                      tensor=tf.reduce_mean(self.online_net_replay_output))
            return self._optimizer.minimize(loss, var_list=self._get_var_list())

    def _store_transition(self, action, observation, reward, terminal):
        if self._replay_scheme == 'uniform':
            priority = 1.
        else:
            priority = self._replay.memory.max_priority()
        self._replay.add(action, observation, reward, terminal, priority)
