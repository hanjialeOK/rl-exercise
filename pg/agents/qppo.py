import numpy as np
import tensorflow as tf

import pg.buffer.gaebuffer as Buffer
from pg.agents.base import BaseAgent
from common.distributions import DiagGaussianPd


def tf_ortho_init(scale):
    return tf.keras.initializers.Orthogonal(scale)


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_dim, name=None):
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

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.squeeze(x, axis=1)


class DiscriminatorMLP(tf.keras.Model):
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


class PPOAgent(BaseAgent):
    def __init__(self, sess, env,
                 clip_ratio=0.2, lr=3e-4, train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=2048, minibatch=64, gamma=0.99, lam=0.95,
                 grad_clip=True, vf_clip=True, fixed_lr=False):
        self.sess = sess
        self.obs_shape = env.observation_space.shape
        self.ac_shape = env.action_space.shape
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
        self.fixed_lr = fixed_lr

        self.buffer = Buffer.QPPOBuffer(
            env, horizon=horizon, gamma=gamma, lam=lam,
            compute_v=self.compute_v)
        self._build_network()
        self._build_train_op()
        super().__init__()

    def _build_network(self):
        self.actor = ActorMLP(self.ac_shape, name='pi')
        self.critic = CriticMLP(name='vf')
        self.discriminator = DiscriminatorMLP(name='q')

    def _build_train_op(self):
        self.ob1_ph = ob1_ph = tf.compat.v1.placeholder(
            shape=(1, ) + self.obs_shape, dtype=tf.float32, name="ob1_ph")
        self.obs_ph = obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_shape, dtype=tf.float32, name="obs_ph")
        self.ac_ph = ac_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.ac_shape, dtype=tf.float32, name="ac_ph")
        self.adv_ph = adv_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="adv_ph")
        self.ret_ph = ret_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="ret_ph")
        self.neglogp_old_ph = neglogp_old_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="neglogp_old_ph")
        self.val_ph = val_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="val_ph")
        self.qval_ph = qval_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="qval_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")

        # Probability distribution
        logstd1 = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.ac_shape[0]),
            initializer=tf.zeros_initializer)

        v1 = self.critic(ob1_ph)

        # Interative with env
        mu1 = self.actor(ob1_ph)
        dist1 = DiagGaussianPd(mu1, logstd1)
        pi1 = dist1.sample()
        neglogp1 = tf.reduce_sum(dist1.neglogp(pi1), axis=1)
        q1 = self.discriminator(ob1_ph, pi1)

        vq = tf.concat([v1, q1], axis=0)

        get_action_ops = [mu1, logstd1, pi1, vq, neglogp1]

        # Train batch data
        mu = self.actor(obs_ph)
        logstd = tf.tile(logstd1, (tf.shape(mu)[0], 1))
        dist = DiagGaussianPd(mu, logstd)
        pi = dist.sample()
        neglogpac = tf.reduce_sum(dist.neglogp(ac_ph), axis=1)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        q_pi = self.discriminator(obs_ph, pi)
        extra_loss = -tf.reduce_mean(q_pi)

        q = self.discriminator(obs_ph, ac_ph)
        meanq = tf.reduce_mean(q)
        q_loss = 0.5 * tf.reduce_mean(tf.square(q - ret_ph))

        # if self.vf_clip:
        #     qvalclipped = qval_ph + \
        #         tf.clip_by_value(q - qval_ph, -self.clip_ratio, self.clip_ratio)
        #     q_loss1 = tf.square(q - ret_ph)
        #     q_loss2 = tf.square(qvalclipped - ret_ph)
        #     q_loss = 0.5 * tf.reduce_mean(tf.maximum(q_loss1, q_loss2))
        # else:
        #     q_loss = 0.5 * tf.reduce_mean(tf.square(q - ret_ph))

        v = self.critic(obs_ph)

        if self.vf_clip:
            valclipped = val_ph + \
                tf.clip_by_value(v - val_ph, -self.clip_ratio, self.clip_ratio)
            vf_loss1 = tf.square(v - ret_ph)
            vf_loss2 = tf.square(valclipped - ret_ph)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))
        else:
            vf_loss = 0.5 * tf.reduce_mean(tf.square(v - ret_ph))

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio = tf.exp(neglogp_old_ph - neglogpac)
        pi_loss1 = -adv_ph * ratio
        pi_loss2 = -adv_ph * tf.clip_by_value(
            ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        pi_loss = tf.reduce_mean(tf.maximum(pi_loss1, pi_loss2))

        # Total loss
        loss = pi_loss - meanent * self.ent_coef + (vf_loss + q_loss) * self.vf_coef

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogp_old_ph - neglogpac))
        absratio = tf.reduce_mean(tf.abs(ratio - 1.0) + 1.0)
        ratioclipped = tf.where(
            adv_ph > 0,
            ratio > (1.0 + self.clip_ratio),
            ratio < (1.0 - self.clip_ratio))
        ratioclipfrac = tf.reduce_mean(tf.cast(ratioclipped, tf.float32))

        # Optimizers
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr_ph, epsilon=1e-5)
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        q_params = self._get_var_list('q')
        params = pi_params + vf_params + q_params
        # grads_and_vars = optimizer.compute_gradients(loss, var_list=params)
        grads_and_vars = optimizer.compute_gradients(loss+extra_loss, var_list=pi_params)
        grads_and_vars += optimizer.compute_gradients(loss, var_list=vf_params+q_params)
        grads, vars = zip(*grads_and_vars)
        if self.grad_clip:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
        gradclipped = _grad_norm > self.max_grad_norm if self.grad_clip else tf.zeros(())
        gradclipped = tf.cast(gradclipped, tf.float32)
        # # Grads correction
        # extra_grads_and_vars = optimizer.compute_gradients(extra_loss, var_list=pi_params)
        # extra_grads, extra_vars = zip(*extra_grads_and_vars)
        # for i in range(7):
        #     grads[i] -= extra_grads[i] * self.max_grad_norm / tf.maximum(_grad_norm, self.max_grad_norm)
        grads_and_vars = list(zip(grads, vars))

        train_op = optimizer.apply_gradients(grads_and_vars)

        self.get_action_ops = get_action_ops
        self.v1 = v1
        self.train_op = train_op

        self.stats_list = [extra_loss, meanq, pi_loss, vf_loss, q_loss, meanent, approxkl, absratio, ratioclipfrac, gradclipped]
        self.loss_names = ['extra_loss', 'q', 'pi_loss', 'vf_loss', 'q_loss', 'entropy', 'kl', 'absratio', 'ratioclipfrac', 'gradclipped']
        assert len(self.stats_list) == len(self.loss_names)

    def update(self, frac, logger):
        buf_data = self.buffer.get()
        [obs_all, ac_all, adv_all, ret_all, val_all, qval_all, neglogp_all] = buf_data
        assert obs_all.shape[0] == self.horizon

        # lr = self.lr if self.fixed_lr else self.lr * frac
        lr = self.lr if self.fixed_lr else np.maximum(self.lr * frac, 1e-4)

        indices = np.arange(self.horizon)

        mblossvals = []
        for _ in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                advs = adv_all[mbinds]
                advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
                inputs = {
                    self.obs_ph: obs_all[mbinds],
                    self.ac_ph: ac_all[mbinds],
                    self.adv_ph: advs,
                    self.ret_ph: ret_all[mbinds],
                    self.neglogp_old_ph: neglogp_all[mbinds],
                    self.val_ph: val_all[mbinds],
                    self.qval_ph: qval_all[mbinds],
                    self.lr_ph: lr,
                }

                losses = self.sess.run(self.stats_list + [self.train_op], feed_dict=inputs)[:-1]
                mblossvals.append(losses)

        # Here you can add any information you want to log!
        lossvals = np.mean(mblossvals, axis=0)
        for (lossval, lossname) in zip(lossvals, self.loss_names):
            logger.logkv('loss/' + lossname, lossval)
        logger.logkv("loss/lr", lr)

    def select_action(self, obs, deterministic=False):
        [mu, logstd, pi, v, neglogp] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        ac = mu if deterministic else pi
        return pi, v, neglogp, mu, logstd

    def compute_v(self, obs):
        v = self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        return v
