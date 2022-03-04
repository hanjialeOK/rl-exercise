import numpy as np
import tensorflow as tf
import os

import pg.buffer.vpgbuffer as Buffer

EPS = 1e-8


def gaussian_likelihood(x, mu, logstd):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(logstd) + EPS)) ** 2 +
                      2 * logstd + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_dim, name=None):
        super(ActorMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.tanh
        kernel_initializer = tf.initializers.orthogonal
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            ac_dim[0],
            kernel_initializer=kernel_initializer, name='fc3')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class CriticMLP(tf.keras.Model):
    def __init__(self, name=None):
        super(CriticMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.tanh
        kernel_initializer = tf.initializers.orthogonal
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
        return x


class PPOAgent():
    def __init__(self, sess, obs_dim, act_dim, clip_ratio=0.2, lr=3e-4,
                 train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=2048, minibatch=64, gamma=0.99, lam=0.95):
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

        self.buffer = Buffer.PPOBuffer(
            obs_dim, act_dim, size=horizon, gamma=gamma, lam=lam)
        self._build_network()
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
        logits = self.actor(obs_ph)
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd',
            initializer=-0.5 * np.ones(self.act_dim, dtype=np.float32))
        std = tf.exp(logstd)
        pi = logits + tf.random.normal(tf.shape(logits)) * std
        logp_a = gaussian_likelihood(act_ph, logits, logstd)
        logp_pi = gaussian_likelihood(pi, logits, logstd)
        entropy = tf.reduce_sum(
            logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

        # State value
        v = tf.compat.v1.squeeze(self.critic(obs_ph), axis=1)

        get_action_ops = [pi, v, logp_pi]

        all_phs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph]

        # PPO objectives
        ratio = tf.exp(logp_a - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.compat.v1.where(adv_ph > 0,
                                     (1 + self.clip_ratio) * adv_ph,
                                     (1 - self.clip_ratio) * adv_ph)
        pi_loss = -tf.reduce_mean(
            tf.minimum(ratio * adv_ph, min_adv))
        v_loss = tf.reduce_mean((ret_ph - v) ** 2)

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        approx_kl = tf.reduce_mean(logp_old_ph - logp_a)
        # a sample estimate for entropy, also easy to compute
        approx_ent = tf.reduce_mean(-logp_a)
        clipped = tf.logical_or(
            ratio > (1 + self.clip_ratio), ratio < (1 - self.clip_ratio))
        clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

        # Total loss
        loss = pi_loss - entropy * self.ent_coef + v_loss * self.vf_coef

        # Optimizers
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.lr, epsilon=1e-5)
        grads_and_vars = optimizer.compute_gradients(loss)
        capped_grads_and_vars = [(tf.clip_by_norm(grad, self.max_grad_norm), var)
                                 for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_grads_and_vars)
        return [obs_ph, all_phs, get_action_ops,
                v, pi_loss, v_loss,
                approx_kl, approx_ent, clipfrac,
                train_op]

    def update(self):
        buf_data = self.buffer.get()

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

                pi_loss_old, v_loss_old, ent = self.sess.run(
                    [self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

                self.sess.run(self.train_op, feed_dict=inputs)

                pi_loss_new, v_loss_new, kl, cf = self.sess.run(
                    [self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac],
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
