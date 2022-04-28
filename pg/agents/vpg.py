import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from pg.agents.base import BaseAgent
import pg.buffer.gaebuffer as Buffer


def tf_ortho_init(scale):
    return tf.keras.initializers.Orthogonal(scale)


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_dim, name=None):
        super(ActorMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.tanh
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            ac_dim[0], activation=None,
            kernel_initializer=tf_ortho_init(0.01), name='fc3')

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
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=tf_ortho_init(1.0), name='fc3')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.squeeze(x, axis=1)


class VPGAgent(BaseAgent):
    """
    The difference between Vanilla Policy Gradient (VPG) with a baseline as value
    function and Advantage Actor-Critic (A2C) is very similar to the difference
    between Monte Carlo Control and SARSA.
    """

    def __init__(self, sess, obs_dim, act_dim,
                 pi_lr=1e-3, vf_lr=1e-3, target_kl=0.01, train_iters=5,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, horizon=2048,
                 minibatch=64, gamma=0.99, lam=0.95, grad_clip=True):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.target_kl = target_kl
        self.train_iters = train_iters
        self.horizon = horizon
        self.minibatch = minibatch
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.grad_clip = grad_clip

        self.buffer = Buffer.GAEBuffer(
            obs_dim, act_dim, size=horizon, gamma=gamma, lam=lam)
        self._build_network()
        self._build_train_op()
        self.saver = self._build_saver()

    def _build_network(self):
        self.actor = ActorMLP(self.act_dim, name='pi')
        self.critic = CriticMLP(name='vf')

    def _build_train_op(self):
        self.ob1_ph = ob1_ph = tf.compat.v1.placeholder(
            shape=(1, ) + self.obs_dim, dtype=tf.float32, name="ob1_ph")
        self.obs_ph = obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        self.act_ph = act_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.act_dim, dtype=tf.float32, name="act_ph")
        self.adv_ph = adv_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="adv_ph")
        self.ret_ph = ret_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="ret_ph")
        self.logp_old_ph = logp_old_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="logp_old_ph")
        self.val_ph = val_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="val_ph")
        self.pi_lr_ph = pi_lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="pi_lr_ph")

        # Probability distribution
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.act_dim[0]),
            initializer=tf.zeros_initializer)
        std = tf.exp(logstd)

        # Interative with env
        mu1 = self.actor(ob1_ph)
        dist1 = tfp.distributions.Normal(loc=mu1, scale=std)
        pi1 = dist1.sample()
        logp_pi1 = tf.reduce_sum(dist1.log_prob(pi1), axis=1)
        v1 = self.critic(ob1_ph)
        get_action_ops = [mu1, pi1, v1, logp_pi1]

        # Train batch data
        mu = self.actor(obs_ph)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        logp_a = tf.reduce_sum(dist.log_prob(act_ph), axis=1)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        v = self.critic(obs_ph)

        # VPG objectives
        pi_loss = -tf.reduce_mean(adv_ph * logp_a)
        vf_loss = 0.5 * tf.reduce_mean(tf.square(v - ret_ph))

        # Usefull Infos
        approx_kl = 0.5 * tf.reduce_mean(tf.square(logp_old_ph - logp_a))

        # Optimizers
        pi_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=pi_lr_ph, epsilon=1e-8)
        vf_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.vf_lr, epsilon=1e-8)
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        # pi_train_op = pi_optimizer.minimize(pi_loss, var_list=pi_params)
        vf_train_op = vf_optimizer.minimize(vf_loss, var_list=vf_params)

        grads_and_vars = pi_optimizer.compute_gradients(
            pi_loss, var_list=pi_params)
        grads, vars = zip(*grads_and_vars)
        if self.grad_clip:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
            is_gradclipped = _grad_norm > self.max_grad_norm
        grads_and_vars = list(zip(grads, vars))
        pi_train_op = pi_optimizer.apply_gradients(grads_and_vars)

        self.get_action_ops = get_action_ops
        self.v1 = v1
        self.pi_loss = pi_loss
        self.vf_loss = vf_loss
        self.entropy = meanent
        self.approx_kl = approx_kl
        self.is_gradclipped = is_gradclipped
        self.pi_train_op = pi_train_op
        self.vf_train_op = vf_train_op

    def update(self, frac=None):
        buf_data = self.buffer.get()
        assert buf_data[0].shape[0] == self.horizon
        [obs, actions, advs, rets, logprobs, values] = buf_data
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

        pi_loss_buf = []
        vf_loss_buf = []
        entropy_buf = []
        kl_buf = []
        lr_buf = []

        pi_inputs = {
            self.obs_ph: obs,
            self.act_ph: actions,
            self.adv_ph: advs,
            self.logp_old_ph: logprobs,
            self.pi_lr_ph: self.pi_lr,
        }

        pi_loss, entropy, klold, is_gradclipped, _ = self.sess.run(
            [self.pi_loss, self.entropy, self.approx_kl,
             self.is_gradclipped, self.pi_train_op],
            feed_dict=pi_inputs)
        klnew = self.sess.run(
            self.approx_kl, feed_dict=pi_inputs)
        pi_loss_buf.append(pi_loss)
        entropy_buf.append(entropy)
        kl_buf.append(klnew)
        lr_buf.append(self.pi_lr)

        if klnew > self.target_kl * 1.5:
            self.pi_lr /= 1.5
        elif klnew < self.target_kl / 1.5:
            self.pi_lr *= 1.5
        # self.pi_lr = np.clip(self.pi_lr, 0, 1e-3)

        indices = np.arange(self.horizon)
        for i in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                vf_inputs = {
                    self.obs_ph: obs[mbinds],
                    self.ret_ph: rets[mbinds]
                }

                vf_loss, _ = self.sess.run(
                    [self.vf_loss, self.vf_train_op],
                    feed_dict=vf_inputs)
                vf_loss_buf.append(vf_loss)

        return [np.mean(pi_loss_buf), np.mean(vf_loss_buf),
                np.mean(entropy_buf), np.mean(kl_buf),
                is_gradclipped, np.mean(lr_buf)]

    def select_action(self, obs, deterministic=False):
        [mu, pi, v, logp_pi] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        self.extra_info = [v, logp_pi]
        ac = mu if deterministic else pi
        return pi[0]

    def compute_v(self, obs):
        v = self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        return v[0]

    def store_transition(self, obs, action, reward, done):
        [v, logp_pi] = self.extra_info
        self.buffer.store(obs, action, reward, done,
                          v[0], logp_pi[0])
