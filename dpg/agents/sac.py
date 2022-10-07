import numpy as np
import tensorflow as tf
from common.distributions import DiagGaussianPd
import os

import dpg.buffer.replaybuffer as Buffer

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_dim, max_ac, name):
        super(ActorMLP, self).__init__(name=name)
        self.ac_limit = max_ac
        # TODO: relu or tanh
        activation_fn = tf.keras.activations.relu
        output_activation_fn = tf.keras.activations.tanh
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            ac_dim[0], activation=None,
            kernel_initializer=kernel_initializer, name='fc3')
        self.dense4 = tf.keras.layers.Dense(
            ac_dim[0], activation=None,
            kernel_initializer=kernel_initializer, name='fc4')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        mu = self.dense3(x)
        logstd = self.dense4(x)
        logstd = tf.clip_by_value(logstd, LOG_STD_MIN, LOG_STD_MAX)
        dist = DiagGaussianPd(mu, logstd)
        pi = dist.sample()
        logp_pi = -tf.reduce_sum(dist.neglogp(pi), axis=1)
        # NOTE: This formula is a little bit magic. To get an understanding of where it
        # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
        # appendix C. This is a more numerically-stable equivalent to Eq 21.
        # Try deriving it yourself as a (very difficult) exercise. :)
        # logp_pi -= tf.reduce_sum(
        #     2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

        # NOTE: 1e-6 work, but 1e-8 do not work.
        logp_pi -= tf.reduce_sum(tf.log(1. - tf.tanh(pi) ** 2 + 1e-6), axis=1)

        # Squash those unbounded actions!
        mu = tf.tanh(mu) * self.ac_limit
        pi = tf.tanh(pi) * self.ac_limit
        return mu, pi, logp_pi


class CriticMLP(tf.keras.Model):
    def __init__(self, name):
        super(CriticMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.relu
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=kernel_initializer, name='fc3')
        self.dense4 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc4')
        self.dense5 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc5')
        self.dense6 = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=kernel_initializer, name='fc6')

    def call(self, state, action):
        x = tf.cast(tf.concat([state, action], axis=-1), tf.float32)
        x1 = self.dense1(x)
        x1 = self.dense2(x1)
        x1 = self.dense3(x1)
        x2 = self.dense4(x)
        x2 = self.dense5(x2)
        x2 = self.dense6(x2)
        return tf.squeeze(x1, axis=1), tf.squeeze(x2, axis=1)


class SACAgent():
    def __init__(self, sess, obs_dim, ac_dim, ac_limit,
                 mem_capacity=int(1e6), batch_size=256,
                 gamma=0.99, actor_lr=3e-4, critic_lr=3e-4,
                 tau=0.005, alpha=0.2):
        self.sess = sess
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.ac_limit = ac_limit
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.alpha = alpha

        self.buffer = Buffer.ReplayBuffer(obs_dim, ac_dim, mem_capacity)
        self._build_network()
        self._build_train_op()

    # Note: Required to be called after _build_train_op(), otherwise return []
    def _get_var_list(self, name='online'):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, name))
        return vars

    def _build_network(self):
        self.actor = ActorMLP(self.ac_dim, self.ac_limit, 'online/actor')
        self.critic = CriticMLP('online/critic')
        self.actor_target = ActorMLP(
            self.ac_dim, self.ac_limit, 'target/actor')
        self.critic_target = CriticMLP('target/critic')

    def _build_train_op(self):
        self.obs_ph = obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        self.next_obs_ph = next_obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="next_obs_ph")
        self.ac_ph = ac_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.ac_dim, dtype=tf.float32, name="ac_ph")
        self.rew_ph = rew_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="rew_ph")
        self.done_ph = done_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="done_ph")

        mu, pi, logp_pi = self.actor(obs_ph)
        q1, q2 = self.critic(obs_ph, pi)
        q = tf.minimum(q1, q2)
        actor_loss = -tf.reduce_mean(q - self.alpha * logp_pi)

        _, pi_next, logp_pi_next = self.actor_target(next_obs_ph)
        target_q1_next, target_q2_next = self.critic_target(
            next_obs_ph, pi_next)
        target_q_next = tf.minimum(target_q1_next, target_q2_next)
        target = rew_ph + (1 - done_ph) * self.gamma * (target_q_next - self.alpha * logp_pi_next)
        target_nograd = tf.stop_gradient(target)

        q1_value, q2_value = self.critic(obs_ph, ac_ph)
        tderr1 = 0.5 * (q1_value - target_nograd) ** 2
        tderr2 = 0.5 * (q2_value - target_nograd) ** 2
        critic_loss = tf.reduce_mean(tderr1 + tderr2)

        actor_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.actor_lr)
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.critic_lr)

        train_actor_op = actor_optimizer.minimize(
            actor_loss, var_list=self._get_var_list('online/actor'))
        train_critic_op = critic_optimizer.minimize(
            critic_loss, var_list=self._get_var_list('online/critic'))

        trainable_online = (self._get_var_list('online/actor')
                            + self._get_var_list('online/critic'))
        trainable_target = (self._get_var_list('target/actor')
                            + self._get_var_list('target/critic'))

        target_update = []
        for (w_online, w_target) in zip(trainable_online, trainable_target):
            w_tau = self.tau * w_online + (1 - self.tau) * w_target
            target_update.append(w_target.assign(w_tau, use_locking=True))

        target_init = []
        for (w_online, w_target) in zip(trainable_online, trainable_target):
            target_init.append(w_target.assign(w_online, use_locking=True))

        self.pi = pi
        self.mu = mu
        self.target_update = target_update
        self.target_init = target_init
        self.train_actor_op = train_actor_op
        self.train_critic_op = train_critic_op

        meanq1 = tf.reduce_mean(q1_value, axis=-1)
        meanq2 = tf.reduce_mean(q2_value, axis=-1)
        self.stats_list = [actor_loss, critic_loss, meanq1, meanq2]
        self.loss_names = ['actor_loss', 'critic_loss', 'q1_value', 'q2_value']
        assert len(self.stats_list) == len(self.loss_names)

    def select_action(self, obs, noise=True):
        get_ac_op = self.pi if noise else self.mu
        pi = self.sess.run(
            get_ac_op, feed_dict={self.obs_ph: obs.reshape(1, -1)})
        return pi[0]

    def target_params_init(self):
        self.sess.run(self.target_init)

    def update(self, logger):
        buf_data = self.buffer.sample_batch(self.batch_size)
        [obs, next_obs, ac, rew, done] = buf_data

        inputs = {
            self.obs_ph: obs,
            self.next_obs_ph: next_obs,
            self.ac_ph: ac,
            self.rew_ph: rew,
            self.done_ph: done
        }

        losses = self.sess.run(
            self.stats_list + [self.train_critic_op, self.train_actor_op], feed_dict=inputs)[:-2]
        # for (lossval, lossname) in zip(losses, self.loss_names):
        #     logger.logkv('loss/' + lossname, lossval)
        # logger.logkv("loss/actor_lr", self.actor_lr)
        # logger.logkv("loss/critic_lr", self.critic_lr)
        # logger.logkv("loss/alpha", self.alpha)
        self.sess.run(self.target_update)

    def bundle(self, checkpoint_dir, iteration):
        if not os.path.exists(checkpoint_dir):
            raise
        self.actor.save_weights(
            os.path.join(checkpoint_dir, 'best_model_actor.h5'), save_format='h5')
        self.critic.save_weights(
            os.path.join(checkpoint_dir, 'best_model_critic.h5'), save_format='h5')
        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration)

    def unbundle(self, checkpoint_dir, iteration=None):
        if not os.path.exists(checkpoint_dir):
            raise
        # Load the best weights without iteraion.
        if iteration is None:
            self.actor.load_weights(
                os.path.join(checkpoint_dir, 'best_model_actor.h5'))
            self.critic.load_weights(
                os.path.join(checkpoint_dir, 'best_model_critic.h5'))
        else:
            self.saver.restore(
                self.sess,
                os.path.join(checkpoint_dir, f'tf_ckpt-{iteration}'))
