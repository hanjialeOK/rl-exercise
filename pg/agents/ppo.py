import numpy as np
import tensorflow as tf
import os
import tensorflow_probability as tfp

import pg.buffer.vpgbuffer as Buffer

from termcolor import cprint


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_dim, name=None):
        super(ActorMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.tanh
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            ac_dim[0],
            kernel_initializer=kernel_initializer, name='fc3')
        self.logstd = tf.compat.v1.get_variable(
            name=os.path.join(name, 'logstd'),
            initializer=-0.5*np.ones(ac_dim, dtype=np.float32))

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        mu = self.dense3(x)
        return mu, self.logstd


class CriticMLP(tf.keras.Model):
    def __init__(self, name=None):
        super(CriticMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.tanh
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            1, kernel_initializer=kernel_initializer, name='fc3')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.squeeze(x, axis=1)


def Actor(obs, ac_dim):
    activation_fn = tf.nn.relu
    kernel_initializer = tf.initializers.orthogonal
    x = tf.compat.v1.layers.dense(
        obs, units=64, activation=activation_fn,
        kernel_initializer=kernel_initializer, name='fc1')
    x = tf.compat.v1.layers.dense(
        x, units=64, activation=activation_fn,
        kernel_initializer=kernel_initializer, name='fc2')
    mu = tf.compat.v1.layers.dense(
        x, units=ac_dim[0], activation=None,
        kernel_initializer=kernel_initializer, name='fc3')
    logstd = tf.compat.v1.get_variable(
        name='logstd',
        initializer=-0.5 * np.ones(ac_dim, dtype=np.float32))
    return mu, logstd


def Critic(obs):
    activation_fn = tf.nn.relu
    kernel_initializer = tf.initializers.orthogonal
    x = tf.compat.v1.layers.dense(
        obs, units=64, activation=activation_fn,
        kernel_initializer=kernel_initializer, name='fc1')
    x = tf.compat.v1.layers.dense(
        x, units=64, activation=activation_fn,
        kernel_initializer=kernel_initializer, name='fc2')
    x = tf.compat.v1.layers.dense(
        x, units=1, activation=None,
        kernel_initializer=kernel_initializer, name='fc3')
    return tf.squeeze(x, axis=1)


class PPOAgent():
    def __init__(self, sess, obs_dim, act_dim, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=3e-4, train_pi_iters=80, train_v_iters=80, target_kl=0.01,
                 horizon=1000, gamma=0.99, lam=0.95, clip_vf=0.1, summary_writer=None,):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.horizon = horizon
        self.clip_vf = clip_vf
        self.summary_writer = summary_writer

        self.buffer = Buffer.PPOBuffer(
            obs_dim, act_dim, size=horizon, gamma=gamma, lam=lam)
        self._build_network()
        [self.obs_ph, self.all_phs, self.get_action_ops,
         self.v, self.pi_loss, self.v_loss,
         self.approx_kl, self.approx_ent,
         self.ratio_clipfrac, self.vf_clipfrac,
         self.train_pi, self.train_v] = self._build_train_op()
        self.saver = self._build_saver()

        if self.summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self.merged_summaries = tf.summary.merge_all()

        self.ratio_clipbox = np.zeros(horizon, dtype=np.float32)
        self.vf_clipbox = np.zeros(horizon, dtype=np.float32)
        self.train_count = 0

    # Note: Required to be called after _build_train_op(), otherwise return []
    def _get_var_list(self, name='pi'):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, name))
        return vars

    def _build_network(self):
        self.actor = ActorMLP(self.act_dim, name='pi')
        self.critic = CriticMLP(name='vf')

    def _build_train_op(self):
        obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        act_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.act_dim, dtype=tf.float32, name="act_ph")
        adv_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="adv_ph")
        ret_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="ret_ph")
        logp_old_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="logp_old_ph")

        # Probability distribution
        mu, logstd = self.actor(obs_ph)
        std = tf.exp(logstd)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        pi = dist.sample()
        logp_a = tf.reduce_sum(dist.log_prob(act_ph), axis=1)
        logp_pi = tf.reduce_sum(dist.log_prob(pi), axis=1)

        # State value
        v = self.critic(obs_ph)

        get_action_ops = [pi, v, logp_pi]

        all_phs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph]

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio = tf.exp(logp_a - logp_old_ph)
        pi_loss1 = adv_ph * ratio
        pi_loss2 = adv_ph * tf.clip_by_value(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        pi_loss = -tf.reduce_mean(tf.minimum(pi_loss1, pi_loss2))

        v_loss = 0.5 * tf.reduce_mean(tf.square(ret_ph - v))

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        # approx_kl = tf.reduce_mean(logp_old_ph - logp_a)
        approx_kl = tf.reduce_mean(ratio - 1 - (logp_a - logp_old_ph))
        # a sample estimate for entropy, also easy to compute
        approx_ent = tf.reduce_mean(-logp_a)
        ratio_clipped = tf.logical_or(
            ratio > (1 + self.clip_ratio), ratio < (1 - self.clip_ratio))
        ratio_clipfrac = tf.reduce_mean(tf.cast(ratio_clipped, tf.float32))
        vf_clipped = (ret_ph - v) > tf.abs(ret_ph * self.clip_vf)
        vf_clipfrac = tf.reduce_mean(tf.cast(vf_clipped, tf.float32))

        if self.summary_writer is not None:
            with tf.variable_scope('pi'):
                pass
            with tf.variable_scope('vf'):
                pass

        # Optimizers
        train_pi = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.pi_lr).minimize(pi_loss)
        train_v = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.vf_lr).minimize(v_loss)

        return [obs_ph, all_phs, get_action_ops,
                v, pi_loss, v_loss,
                approx_kl, approx_ent,
                ratio_clipfrac, vf_clipfrac,
                train_pi, train_v]

    def update(self):
        self.train_count += 1

        buf_data = self.buffer.get()
        inputs = {k: v for k, v in zip(self.all_phs, buf_data)}

        pi_loss_old, v_loss_old, ent = self.sess.run(
            [self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

        for i in range(self.train_pi_iters):
            _, kl = self.sess.run(
                [self.train_pi, self.approx_kl], feed_dict=inputs)
            if kl > 1.5 * self.target_kl:
                cprint(f'Early stopping at step {i} due to reaching max kl.',
                       color='green', attrs=['bold'])
                break
        for _ in range(self.train_v_iters):
            ratio_clipf, vf_clipf, _ = self.sess.run(
                [self.ratio_clipfrac, self.vf_clipfrac, self.train_v],
                feed_dict=inputs)
            self.ratio_clipbox[self.train_count - 1] = ratio_clipf
            self.vf_clipbox[self.train_count - 1] = vf_clipf

        vf_summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag="vf/clipfrac", simple_value=np.mean(self.vf_clipbox)),
            tf.compat.v1.Summary.Value(
                tag="ratio/clipfrac", simple_value=np.mean(self.ratio_clipbox))
        ])
        self.summary_writer.add_summary(vf_summary, self.train_count)

        pi_loss_new, v_loss_new, kl = self.sess.run(
            [self.pi_loss, self.v_loss, self.approx_kl],
            feed_dict=inputs)

    def _build_saver(self):
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        return tf.compat.v1.train.Saver(var_list=pi_params + vf_params,
                                        max_to_keep=4)

    def select_action(self, obs):
        [pi, v_t, logp_pi_t] = self.sess.run(
            self.get_action_ops, feed_dict={self.obs_ph: obs.reshape(1, -1)})
        self.extra_info = [v_t, logp_pi_t]
        return pi[0]

    def compute_v(self, obs):
        return self.sess.run(
            self.v, feed_dict={self.obs_ph: obs.reshape(1, -1)})

    def store_transition(self, obs, action, reward, done):
        [v_t, logp_pi_t] = self.extra_info
        self.buffer.store(obs, action, reward, done,
                          v_t, logp_pi_t)

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
