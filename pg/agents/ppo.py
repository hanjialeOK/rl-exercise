import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from pg.agents.base import BaseAgent
import pg.buffer.gaebuffer as Buffer


def tf_ortho_init(scale):
    return tf.keras.initializers.Orthogonal(scale)


class PPOAgent(BaseAgent):
    def __init__(self, sess, obs_dim, act_dim, num_env=1,
                 clip_ratio=0.2, lr=3e-4, train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=2048, minibatch=64, gamma=0.99, lam=0.95,
                 grad_clip=True, vf_clip=True, fixed_lr=False):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_env = num_env
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.horizon = horizon
        self.minibatch = minibatch * num_env
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.grad_clip = grad_clip
        self.vf_clip = vf_clip
        self.fixed_lr = fixed_lr

        self.buffer = Buffer.GAEBuffer(
            obs_dim, act_dim, size=horizon, num_env=num_env, gamma=gamma, lam=lam)
        self._build_train_op()
        self.saver = self._build_saver()

    def Actor(self, obs):
        activation_fn = tf.tanh
        kernel_initializer = None
        with tf.compat.v1.variable_scope('pi', reuse=tf.AUTO_REUSE):
            x = tf.compat.v1.layers.dense(
                obs, units=64, activation=activation_fn,
                kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc1')
            x = tf.compat.v1.layers.dense(
                x, units=64, activation=activation_fn,
                kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc2')
            x = tf.compat.v1.layers.dense(
                x, units=self.act_dim[0], activation=None,
                kernel_initializer=tf_ortho_init(0.01), name='fc3')
            return x

    def Critic(self, obs):
        activation_fn = tf.tanh
        kernel_initializer = None
        with tf.compat.v1.variable_scope('vf', reuse=tf.AUTO_REUSE):
            x = tf.compat.v1.layers.dense(
                obs, units=64, activation=activation_fn,
                kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc1')
            x = tf.compat.v1.layers.dense(
                x, units=64, activation=activation_fn,
                kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc2')
            x = tf.compat.v1.layers.dense(
                x, units=1, activation=None,
                kernel_initializer=tf_ortho_init(1), name='fc3')
            return tf.squeeze(x, axis=1)

    def _build_train_op(self):
        ob1_ph = tf.compat.v1.placeholder(
            shape=(self.num_env, ) + self.obs_dim, dtype=tf.float32, name="ob1_ph")
        obs_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        act_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.act_dim, dtype=tf.float32, name="act_ph")
        adv_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="adv_ph")
        ret_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="ret_ph")
        logp_old_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="logp_old_ph")
        val_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="val_ph")
        lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")

        all_phs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph, val_ph, lr_ph]

        # Probability distribution
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.act_dim[0]),
            initializer=tf.zeros_initializer)
        std = tf.exp(logstd)

        # Interative with env
        mu1 = self.Actor(ob1_ph)
        dist1 = tfp.distributions.Normal(loc=mu1, scale=std)
        pi1 = dist1.sample()
        logp_pi1 = tf.reduce_sum(dist1.log_prob(pi1), axis=1)

        v1 = self.Critic(ob1_ph)

        get_action_ops = [mu1, pi1, v1, logp_pi1]

        # Train batch data
        mu = self.Actor(obs_ph)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        logp_a = tf.reduce_sum(dist.log_prob(act_ph), axis=1)
        entropy = tf.reduce_mean(dist.entropy())

        v = self.Critic(obs_ph)

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio = tf.exp(logp_a - logp_old_ph)
        pi_loss1 = -adv_ph * ratio
        pi_loss2 = -adv_ph * tf.clip_by_value(
            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        pi_loss = tf.reduce_mean(tf.maximum(pi_loss1, pi_loss2))

        if self.vf_clip:
            valclipped = val_ph + \
                tf.clip_by_value(v - val_ph, -self.clip_ratio, self.clip_ratio)
            vf_loss1 = tf.square(v - ret_ph)
            vf_loss2 = tf.square(valclipped - ret_ph)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
        else:
            vf_loss = 0.5 * tf.reduce_mean(tf.square(v - ret_ph))

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        approx_kl = 0.5 * tf.reduce_mean(tf.square(logp_old_ph - logp_a))
        clipped = tf.logical_or(
            ratio > (1.0 + self.clip_ratio), ratio < (1.0 - self.clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Total loss
        loss = pi_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        # Optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph, epsilon=1e-5)
        params = self._get_var_list('pi') + self._get_var_list('vf')
        grads_and_vars = optimizer.compute_gradients(loss, var_list=params)
        grads, vars = zip(*grads_and_vars)
        if self.grad_clip:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
        grads_and_vars = list(zip(grads, vars))

        train_op = optimizer.apply_gradients(grads_and_vars)

        self.ob1_ph = ob1_ph
        self.all_phs = all_phs
        self.get_action_ops = get_action_ops
        self.v1 = v1
        self.pi_loss = pi_loss
        self.vf_loss = vf_loss
        self.approx_kl = approx_kl
        self.entropy = entropy
        self.clipfrac = clipfrac
        self.train_op = train_op

    def update(self, frac):
        buf_data = self.buffer.get()
        assert buf_data[0].shape[0] == self.horizon * self.num_env

        pi_loss_buf = []
        vf_loss_buf = []
        entropy_buf = []
        kl_buf = []

        indices = np.arange(self.horizon * self.num_env)
        for _ in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                slices = [arr[mbinds] for arr in buf_data]
                [obs, actions, advs, rets, logprobs, values] = slices
                # advs_raw = rets - values
                advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
                lr = self.lr if self.fixed_lr else self.lr * frac
                inputs = {
                    self.all_phs[0]: obs,
                    self.all_phs[1]: actions,
                    self.all_phs[2]: advs,
                    self.all_phs[3]: rets,
                    self.all_phs[4]: logprobs,
                    self.all_phs[5]: values,
                    self.all_phs[6]: lr,
                }

                pi_loss, vf_loss, entropy, kl, _ = self.sess.run(
                    [self.pi_loss, self.vf_loss,
                     self.entropy, self.approx_kl,
                     self.train_op],
                    feed_dict=inputs)
                pi_loss_buf.append(pi_loss)
                vf_loss_buf.append(vf_loss)
                entropy_buf.append(entropy)
                kl_buf.append(kl)

        return [np.mean(pi_loss_buf), np.mean(vf_loss_buf),
                np.mean(entropy_buf), np.mean(kl_buf)]

    def select_action(self, obs, deterministic=False):
        [mu, pi, v, logp_pi] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(self.num_env, -1)})
        self.extra_info = [v, logp_pi]
        ac = mu if deterministic else pi
        return pi

    def compute_v(self, obs):
        return self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs.reshape(self.num_env, -1)})

    def store_transition(self, obs, action, reward, done):
        [v, logp_pi] = self.extra_info
        self.buffer.store(obs, action, reward, done,
                          v, logp_pi)
