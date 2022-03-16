import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

import pg.buffer.vpgbuffer as Buffer


def Actor(obs, ac_dim):
    activation_fn = tf.tanh
    kernel_initializer = None
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
    activation_fn = tf.tanh
    kernel_initializer = None
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
    def __init__(self, sess, obs_dim, act_dim, clip_ratio=0.2, lr=3e-4,
                 train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=2048, minibatch=64, gamma=0.99, lam=0.95,
                 grad_clip=True, vf_clip=True):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.horizon = horizon
        self.minibatch = minibatch
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.grad_clip = grad_clip
        self.vf_clip = vf_clip

        self.buffer = Buffer.PPOBuffer(
            obs_dim, act_dim, size=horizon, gamma=gamma, lam=lam)
        [self.obs_ph, self.all_phs, self.get_action_ops,
         self.v, self.pi_loss, self.v_loss,
         self.approx_kl, self.approx_ent, self.clipfrac,
         self.train_op] = self._build_train_op()
        self.saver = self._build_saver()

    # Note: Required to be called after _build_train_op(), otherwise return []
    def _get_var_list(self, name='pi'):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, name))
        return vars

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
        val_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="val_ph")

        # Probability distribution
        with tf.variable_scope('pi'):
            mu, logstd = Actor(obs_ph, self.act_dim)
            std = tf.exp(logstd)
            dist = tfp.distributions.Normal(loc=mu, scale=std)
            pi = dist.sample()
            logp_a = tf.reduce_sum(dist.log_prob(act_ph), axis=1)
            logp_pi = tf.reduce_sum(dist.log_prob(pi), axis=1)
            entropy = tf.reduce_mean(dist.entropy())

        # State value
        with tf.variable_scope('v'):
            v = Critic(obs_ph)

        get_action_ops = [mu, pi, v, logp_pi]

        all_phs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph, val_ph]

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio = tf.exp(logp_a - logp_old_ph)
        pi_loss1 = adv_ph * ratio
        pi_loss2 = adv_ph * tf.clip_by_value(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        pi_loss = -tf.reduce_mean(tf.minimum(pi_loss1, pi_loss2))

        if self.vf_clip:
            valclipped = val_ph + \
                tf.clip_by_value(v - val_ph, -self.clip_ratio, self.clip_ratio)
            v_loss1 = tf.square(v - ret_ph)
            v_loss2 = tf.square(valclipped - ret_ph)
            v_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2))
        else:
            v_loss = 0.5 * tf.reduce_mean(tf.square(v - ret_ph))

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        approx_kl = 0.5 * tf.reduce_mean(tf.square(logp_old_ph - logp_a))
        clipped = tf.logical_or(
            ratio > (1 + self.clip_ratio), ratio < (1 - self.clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Total loss
        loss = pi_loss - entropy * self.ent_coef + v_loss * self.vf_coef

        # Optimizers
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.lr, epsilon=1e-5)
        if self.grad_clip:
            grads_and_vars = optimizer.compute_gradients(loss)
            capped_grads_and_vars = [(tf.clip_by_norm(grad, self.max_grad_norm), var)
                                     for grad, var in grads_and_vars]
            train_op = optimizer.apply_gradients(capped_grads_and_vars)
        else:
            train_op = optimizer.minimize(loss)
        return [obs_ph, all_phs, get_action_ops,
                v, pi_loss, v_loss,
                approx_kl, entropy, clipfrac,
                train_op]

    def update(self):
        buf_data = self.buffer.get()

        pi_loss_buf = []
        v_loss_buf = []
        entropy_buf = []
        kl_buf = []

        indices = np.arange(self.horizon)
        for _ in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                slices = [arr[mbinds] for arr in buf_data]
                inputs = {k: v for k, v in zip(self.all_phs, slices)}

                pi_loss, v_loss, entropy, kl, _ = self.sess.run(
                    [self.pi_loss, self.v_loss,
                     self.approx_ent, self.approx_kl,
                     self.train_op],
                    feed_dict=inputs)
                pi_loss_buf.append(pi_loss)
                v_loss_buf.append(v_loss)
                entropy_buf.append(entropy)
                kl_buf.append(kl)

        return [np.mean(pi_loss_buf), np.mean(v_loss_buf),
                np.mean(entropy_buf), np.mean(kl_buf)]

    def _build_saver(self):
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('v')
        return tf.compat.v1.train.Saver(var_list=pi_params + vf_params,
                                        max_to_keep=4)

    def select_action(self, obs, deterministic=False):
        [mu, pi, v_t, logp_pi_t] = self.sess.run(
            self.get_action_ops, feed_dict={self.obs_ph: obs.reshape(1, -1)})
        self.extra_info = [v_t, logp_pi_t]
        ac = mu[0] if deterministic else pi[0]
        return ac

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
        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration)

    def unbundle(self, checkpoint_dir, iteration=None):
        if not os.path.exists(checkpoint_dir):
            raise
        self.saver.restore(
            self.sess,
            os.path.join(checkpoint_dir, f'tf_ckpt-{iteration}'))
