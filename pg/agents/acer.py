import numpy as np
import tensorflow as tf

import pg.buffer.gaebuffer as Buffer
from pg.agents.base import BaseAgent
from common.distributions import DiagGaussianPd


def tf_ortho_init(scale):
    return tf.keras.initializers.Orthogonal(scale)

def q_retrace(r, d, q, v, rho, last_v, gamma, ac_dim, minibatch):
    rho_bar = tf.minimum(1.0, tf.pow(rho, 1/ac_dim))
    lastqret = last_v
    lastqopc = last_v
    qrets = []
    qopcs = []
    for i in reversed(range(0, minibatch)):
        nondone = 1.0 - d[i]
        qret = r[i] + gamma * lastqret * nondone
        qopc = r[i] + gamma * lastqopc * nondone
        qrets.append(qret)
        qopcs.append(qopc)
        lastqret = rho_bar[i] * (qret - q[i]) + v[i]
        lastqopc = (qopc - q[i]) + v[i]
    qrets = qrets[::-1]
    qopcs = qopcs[::-1]
    qrets = tf.concat(qrets, axis=0)
    qopcs = tf.concat(qopcs, axis=0)
    return qrets, qopcs


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_shape, name=None):
        super().__init__(name=name)
        activation_fn = tf.keras.activations.tanh
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            ac_shape[0], activation=None,
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


class ADVMLP(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
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

    def call(self, state, action):
        x = tf.cast(tf.concat([state, action], axis=-1), tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.squeeze(x, axis=1)


class ACERAgent(BaseAgent):
    def __init__(self, sess, env,
                 lr=3e-4, ent_coef=0.0, q_coef=0.5, max_grad_norm=10,
                 horizon=50, gamma=0.99, std_init=0.3, replay_init=1000,
                 grad_clip=True, n_sdn_samples=5, replay_size=5000, decay=0.995,
                 replay_ratio=4, c=5, trust_region=True, delta=0.1, fixed_lr=True):
        self.sess = sess
        self.obs_shape = env.observation_space.shape
        self.ac_shape = env.action_space.shape
        self.lr = lr
        self.horizon = horizon
        self.minibatch = horizon
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.q_coef = q_coef
        self.max_grad_norm = max_grad_norm
        self.grad_clip = grad_clip
        self.fixed_lr = fixed_lr
        self.n_sdn_samples = n_sdn_samples
        self.replay_size = replay_size
        self.decay = decay
        self.replay_ratio = replay_ratio
        self.c = c
        self.trust_region = trust_region
        self.delta = delta
        self.std_init= std_init
        self.replay_init = replay_init

        self.buffer = Buffer.ACERBuffer(
            env, horizon=horizon, replay_size=replay_size, gamma=gamma)
        self._build_network()
        self._build_train_op()
        self.init_avg_op = self._build_sync_op()
        self.sync_avg_op = self._build_average_op()
        super().__init__()

    def _build_network(self):
        self.actor = ActorMLP(self.ac_shape, name='pi')
        self.actor_a = ActorMLP(self.ac_shape, name='pi_a')
        self.critic = CriticMLP('vf')
        self.advantage = ADVMLP(name='adv')

    def sdn_forword(self, state, action, action_samples):
        v = self.critic(state)
        adv = self.advantage(state, action)
        adv_samples = [self.advantage(state, ac_i) for ac_i in action_samples]
        adv_samples = tf.stack(adv_samples)
        q = v + adv - tf.reduce_mean(adv_samples, axis=0)
        return v, q

    def _build_train_op(self):
        self.ob1_ph = ob1_ph = tf.compat.v1.placeholder(
            shape=(1, ) + self.obs_shape, dtype=tf.float32, name="ob1_ph")
        self.obs_ph = obs_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch+1, ) + self.obs_shape, dtype=tf.float32, name="obs_ph")
        self.ac_ph = ac_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.ac_shape, dtype=tf.float32, name="ac_ph")
        self.done_ph = done_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="done_ph")
        self.rew_ph = rew_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="rew_ph")
        self.neglogp_old_ph = neglogp_old_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ), dtype=tf.float32, name="neglogp_old_ph")
        self.mu_old_ph = mu_old_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.ac_shape, dtype=tf.float32, name="mu_old_ph")
        self.logstd_old_ph = logstd_old_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.ac_shape, dtype=tf.float32, name="logstd_old_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")

        # Fixed standard deviation
        logstd1 = tf.ones((1, self.ac_shape[0]), dtype=tf.float32) * tf.log(self.std_init)
        logstd1_a = tf.ones((1, self.ac_shape[0]), dtype=tf.float32) * tf.log(self.std_init)

        # Interative with env
        mu1 = self.actor(ob1_ph)
        dist1 = DiagGaussianPd(mu1, logstd1)
        pi1 = dist1.sample()
        neglogp1 = tf.reduce_sum(dist1.neglogp(pi1), axis=1)
        v1 = self.critic(ob1_ph)
        get_action_ops = [mu1, logstd1, pi1, v1, neglogp1]

        # Train batch data
        mu = self.actor(obs_ph[:-1])
        logstd = tf.tile(logstd1, (tf.shape(mu)[0], 1))
        dist = DiagGaussianPd(mu, logstd)
        neglogpac = tf.reduce_sum(dist.neglogp(ac_ph), axis=1)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        # Average actor
        mu_a = tf.stop_gradient(self.actor_a(obs_ph[:-1]))
        logstd_a = tf.tile(logstd1_a, (tf.shape(mu_a)[0], 1))
        dist_a = DiagGaussianPd(mu_a, logstd_a)
        kl = tf.reduce_sum(dist_a.kl(dist), axis=1)
        meankl = tf.reduce_mean(kl)

        # Q_Retrace
        ac_samples = [dist.sample() for _ in range(self.n_sdn_samples)]
        v, q = self.sdn_forword(obs_ph[:-1], ac_ph, ac_samples)
        rho = tf.exp(neglogp_old_ph - neglogpac)
        last_v = self.critic(tf.reshape(obs_ph[-1], (1, -1)))
        qret, qopc = q_retrace(rew_ph, done_ph, q, v, rho, last_v, self.gamma, self.ac_shape[0], self.minibatch)
        v_targ = tf.minimum(rho, 1.0) * (qret - q) + v

        # Alternative action
        ac2 = dist.sample()
        neglogpac2 = tf.reduce_sum(dist.neglogp(ac2), axis=1)
        ac2_samples = [dist.sample() for _ in range(self.n_sdn_samples)]
        _, q2 = self.sdn_forword(obs_ph[:-1], ac2, ac2_samples)

        # rho and alternative rho
        dist_old = DiagGaussianPd(mu_old_ph, logstd_old_ph)
        neglogpac2_old = tf.reduce_sum(dist_old.neglogp(ac2), axis=1)
        rho2 = tf.exp(neglogpac2_old - neglogpac2)

        # A2C objectives
        pi_loss = -tf.stop_gradient(tf.minimum(rho, self.c) * (qopc - v)) * (-neglogpac)
        pi_loss_bc = -tf.stop_gradient(tf.nn.relu(1 - self.c / rho2) * (q2 - v)) * (-neglogpac2)
        pi_loss += pi_loss_bc
        pi_loss = tf.reduce_mean(pi_loss)
        q_loss = 0.5 * tf.square(tf.stop_gradient(qret) - q)
        vf_loss = 0.5 * tf.square(tf.stop_gradient(v_targ) - v)
        q_loss += vf_loss
        q_loss = tf.reduce_mean(q_loss)

        # Total loss
        loss = pi_loss - meanent * self.ent_coef + q_loss * self.q_coef

        pi_params = self._get_var_list('pi')
        q_params = self._get_var_list('vf') + self._get_var_list('adv')
        if self.trust_region:
            phi = [mu, logstd]
            # It's very important to multiply by minibatch because k and g are both mean grads.
            g = tf.compat.v1.gradients(ys=-(pi_loss - meanent * self.ent_coef)*self.minibatch, xs=phi)
            k = tf.compat.v1.gradients(ys=meankl*self.minibatch, xs=phi)
            grads_phi = []
            k_dot_g_all = []
            for g_i, k_i in zip(g, k):
                k_dot_g = tf.reduce_sum(k_i * g_i, axis=-1)
                k_dot_g_all.append(k_dot_g)
                adj = tf.maximum(0.0, tf.div(k_dot_g - self.delta, tf.reduce_sum(tf.square(k_i), axis=-1)))
                g_i -= tf.expand_dims(adj, axis=-1) * k_i
                # Dont't forget to divide by self.minibatch for mean grads.
                g_i = -g_i / self.minibatch
                grads_phi.append(g_i)
            grads_pi = tf.compat.v1.gradients(phi, pi_params, grads_phi)
            grads_q = tf.compat.v1.gradients(q_loss * self.q_coef, q_params)
            grads = grads_pi + grads_q
        else:
            grads = tf.compat.v1.gradients(loss, pi_params+q_params)

        # Optimizers
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr_ph, epsilon=1e-5)
        if self.grad_clip:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
        gradclipped = _grad_norm > self.max_grad_norm if self.grad_clip else tf.zeros(())
        grads_and_vars = list(zip(grads, pi_params+q_params))

        train_op = optimizer.apply_gradients(grads_and_vars)

        self.get_action_ops = get_action_ops
        self.v1 = v1
        self.neglogpac = neglogpac
        self.v = v
        self.q = q
        self.pi_loss = pi_loss
        self.q_loss = q_loss
        self.entropy = meanent
        self.gradclipped = gradclipped
        self.train_op = train_op

        self.losses = [pi_loss_bc, pi_loss, q_loss, meanent, meankl]
        self.infos = [gradclipped, k_dot_g_all]

    def _build_average_op(self):
        sync_avg_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_a = self._get_var_list('pi_a')

        for (var_a, var) in zip(pi_params_a, pi_params):
            var_a_new = self.decay * var_a + (1 - self.decay) * var
            sync_avg_ops.append(var_a.assign(var_a_new, use_locking=True))
        return sync_avg_ops

    def _build_sync_op(self):
        sync_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_a = self._get_var_list('pi_a')

        for (var_a, var) in zip(pi_params_a, pi_params):
            sync_ops.append(var_a.assign(var, use_locking=True))
        return sync_ops

    def update(self, frac, logger):
        
        pi_loss_rc_buf = []
        pi_loss_buf = []
        q_loss_buf = []
        ent_buf = []
        kl_buf = []
        gradclipped_buf = []

        if self.buffer.count < self.replay_init:
            return 0, 0, 0, 0

        lr = self.lr if self.fixed_lr else self.lr * frac

        n_replay_samples = np.random.poisson(self.replay_ratio)
        # n_replay_samples = 4
        for _ in range(n_replay_samples):
            buf_data = self.buffer.get()
            [obs, ac, rew, done, neglogp, mean, logstd] = buf_data
            inputs = {
                self.obs_ph: obs,
                self.ac_ph: ac,
                self.done_ph: done,
                self.rew_ph: rew,
                self.neglogp_old_ph: neglogp,
                self.mu_old_ph: mean,
                self.logstd_old_ph: logstd,
                self.lr_ph: lr
            }

            infos, losses, _ = self.sess.run(
                [self.infos, self.losses, self.train_op],
                feed_dict=inputs)
            self.sess.run(self.sync_avg_op)

            # Unpack losses
            pi_loss_rc, pi_loss, q_loss, ent, kl = losses
            pi_loss_rc_buf.append(pi_loss_rc)
            pi_loss_buf.append(pi_loss)
            q_loss_buf.append(q_loss)
            ent_buf.append(ent)
            kl_buf.append(kl)
            # Unpack infos
            [gradclipped] = infos
            gradclipped_buf.append(gradclipped)

        # Here you can add any information you want to log!
        logger.logkv("loss/n_replay_samples", n_replay_samples)
        logger.logkv("loss/pi_loss_rc", np.mean(pi_loss_rc_buf))
        logger.logkv("loss/q_loss", np.mean(q_loss_buf))
        logger.logkv("loss/gradclipfrac", np.mean(gradclipped_buf))
        logger.logkv("loss/lr", lr)

        return [np.mean(pi_loss_buf), np.mean(q_loss_buf),
                np.mean(ent_buf), np.mean(kl_buf)]

    def select_action(self, obs, deterministic=False):
        [mu, logstd, pi, v, neglogp] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        ac = mu if deterministic else pi
        return pi, v, neglogp, mu, logstd

    def compute_v(self, obs):
        v = self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        return v
