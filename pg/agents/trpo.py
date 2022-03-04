import numpy as np
import tensorflow as tf
import os

import pg.buffer.vpgbuffer as Buffer

from termcolor import cprint

EPS = 1e-8


def cg(Ax, b, cg_iters):
    """
    Conjugate gradient algorithm
    """
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    r_dot_old = np.dot(r, r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def flat_grad(f, params):
    return flat_concat(tf.compat.v1.gradients(ys=f, xs=params))


def assign_params_from_flat(new_params_flated, params):
    # the 'int' is important for scalars
    def flat_size(p): return int(np.prod(p.shape.as_list()))
    splits = tf.split(new_params_flated, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape)
                  for p_new, p in zip(splits, params)]
    return tf.group(
        [tf.compat.v1.assign(p, p_new) for p, p_new in zip(params, new_params)])


def gaussian_likelihood(x, mu, logstd):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(logstd) + EPS)) ** 2 +
                      2 * logstd + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def diagonal_gaussian_kl(mu0, logstd0, mu1, logstd1):
    """
    tf symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    var0, var1 = tf.exp(2 * logstd0), tf.exp(2 * logstd1)
    pre_sum = 0.5*(((mu1 - mu0)**2 + var0)/(var1 + EPS) - 1) + \
        logstd1 - logstd0
    all_kls = tf.reduce_sum(pre_sum, axis=1)
    return tf.reduce_mean(all_kls)


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


class TRPOAgent():
    def __init__(self, sess, obs_dim, act_dim, max_kl=0.01, vf_lr=1e-3,
                 train_v_iters=80, damping_coeff=0.1, cg_iters=10,
                 backtrack_iters=10, backtrack_coeff=0.8,
                 horizon=1000, gamma=0.99, lam=0.95):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_kl = max_kl
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.horizon = horizon

        self.buffer = Buffer.TRPOBuffer(
            obs_dim, act_dim, size=horizon, gamma=gamma, lam=lam)
        self._build_network()
        [self.obs_ph, self.all_phs, self.x_ph, self.param_ph,
         self.get_action_ops, self.d_kl,
         self.pi_grads_flatted, self.pi_params_flatted, self.set_pi_params,
         self.hvp, self.pi_loss, self.v, self.train_vf] = self._build_train_op()
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
        old_logits_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.act_dim, dtype=tf.float32, name='old_logits_ph')
        old_logstd_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.act_dim, dtype=tf.float32, name='old_logstd_ph')

        # Probability distribution
        logits = self.actor(obs_ph)
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd',
            initializer=-0.5 * np.ones(self.act_dim, dtype=np.float32))
        std = tf.exp(logstd)
        pi = logits + tf.compat.v1.random_normal(tf.shape(logits)) * std
        logp = gaussian_likelihood(act_ph, logits, logstd)
        logp_pi = gaussian_likelihood(pi, logits, logstd)
        d_kl = diagonal_gaussian_kl(
            logits, logstd, old_logits_ph, old_logstd_ph)

        # State value
        v = tf.compat.v1.squeeze(self.critic(obs_ph), axis=1)

        get_action_ops = [pi, v, logp_pi, logits, logstd]

        all_phs = [obs_ph, act_ph, adv_ph, ret_ph,
                   logp_old_ph, old_logits_ph, old_logstd_ph]

        # TRPO losses
        ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        pi_loss = -tf.reduce_mean(ratio * adv_ph)
        v_loss = tf.reduce_mean((ret_ph - v)**2)

        # Optimizer for value funtion
        train_vf = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.vf_lr).minimize(v_loss)

        # Symbols needed for CG solver
        pi_params = self._get_var_list('pi')
        pi_grads_flatted = flat_grad(pi_loss, pi_params)
        kl_grads_flatted = flat_grad(d_kl, pi_params)
        x_ph = tf.compat.v1.placeholder(shape=kl_grads_flatted.shape,
                                        dtype=tf.float32, name='x_ph')
        hvp = flat_grad(
            tf.reduce_sum(kl_grads_flatted * x_ph), pi_params)
        if self.damping_coeff > 0:
            hvp += self.damping_coeff * x_ph

        # Symbols for getting and setting params
        pi_params_flatted = flat_concat(pi_params)
        param_ph = tf.compat.v1.placeholder(
            shape=pi_params_flatted.shape,
            dtype=tf.float32, name="param_ph")
        set_pi_params = assign_params_from_flat(
            param_ph, pi_params)

        return [obs_ph, all_phs, x_ph, param_ph,
                get_action_ops, d_kl,
                pi_grads_flatted,
                pi_params_flatted, set_pi_params,
                hvp, pi_loss, v, train_vf]

    def update(self):
        # Prepare hessian func, gradient eval
        buf_data = self.buffer.get()
        inputs = {k: v for k, v in zip(self.all_phs, buf_data)}

        def Hx(x): return self.sess.run(
            self.hvp, feed_dict={**inputs, self.x_ph: x})

        g, pi_loss_old = self.sess.run(
            [self.pi_grads_flatted, self.pi_loss], feed_dict=inputs)

        # Core calculations for TRPO or NPG
        x = cg(Hx, g, self.cg_iters)
        alpha = np.sqrt(2 * self.max_kl / (np.dot(x, Hx(x)) + EPS))
        old_params = self.sess.run(self.pi_params_flatted)

        for j in range(self.backtrack_iters):
            step = self.backtrack_coeff ** j
            self.sess.run(self.set_pi_params,
                          feed_dict={self.param_ph: old_params - alpha * x * step})
            kl, pi_loss_new = self.sess.run(
                [self.d_kl, self.pi_loss], feed_dict=inputs)
            if kl < self.max_kl and pi_loss_new < pi_loss_old:
                cprint(f'Accepting new params at step {j} of line search.',
                       color='green', attrs=['bold'])
                break
            if j == self.backtrack_iters - 1:
                cprint('Line search failed! Keeping old params.',
                       color='red', attrs=['bold'])
                self.sess.run(self.set_pi_params,
                              feed_dict={self.param_ph: old_params})

        for _ in range(self.train_v_iters):
            self.sess.run(self.train_vf, feed_dict=inputs)

    def _build_saver(self):
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        return tf.compat.v1.train.Saver(var_list=pi_params + vf_params,
                                        max_to_keep=4)

    def select_action(self, obs):
        [pi, v_t, logp_pi_t, logits_t, logstd_t] = self.sess.run(
            self.get_action_ops, feed_dict={self.obs_ph: obs.reshape(1, -1)})
        self.extra_info = [v_t, logp_pi_t, logits_t, logstd_t]
        return pi[0]

    def compute_v(self, obs):
        return self.sess.run(
            self.v, feed_dict={self.obs_ph: obs.reshape(1, -1)})

    def store_transition(self, obs, action, reward, done):
        [v_t, logp_pi_t, logits_t, logstd_t] = self.extra_info
        self.buffer.store(obs, action, reward, done,
                          v_t, logp_pi_t, logits_t, logstd_t)

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
