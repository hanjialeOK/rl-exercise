import numpy as np
import tensorflow as tf

from deepq.agents.dqn import DQNAgent


class DDQNAgent(DQNAgent):
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
        super(DDQNAgent, self).__init__(
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

    def _build_target_q_op(self):
        online_next_q = self.online_network(self.replay_next_states)  # (32, 4)
        replay_actions_one_hot = tf.one_hot(
            tf.argmax(online_next_q, axis=1), self._num_actions, 1., 0., axis=-1)  # (32, 4)
        next_q = tf.reduce_sum(
            self.target_net_replay_output * replay_actions_one_hot, axis=1)  # (32,)
        target = self.replay_rewards + self._gamma * \
            (1 - self.replay_terminals) * next_q  # (32,)
        return target
