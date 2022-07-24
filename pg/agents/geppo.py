import numpy as np
import tensorflow as tf

import pg.buffer.gaebuffer as Buffer
from pg.agents.base import BaseAgent
from common.distributions import DiagGaussianPd


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


class PPOAgent(BaseAgent):
    def __init__(self, sess, env,
                 clip_ratio=0.1, lr=3e-4, train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=1024, nminibatches=32, gamma=0.99, lam=0.95,
                 grad_clip=True, vf_clip=False, fixed_lr=False,
                 thresh=0.5, alpha=0.03, nlatest=4, uniform=False):
        self.sess = sess
        self.obs_shape = env.observation_space.shape
        self.ac_shape = env.action_space.shape
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.horizon = horizon
        self.nminibatches = nminibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.grad_clip = grad_clip
        self.vf_clip = vf_clip
        self.fixed_lr = fixed_lr
        self.thresh = thresh
        self.alpha = alpha
        self.nlatest = nlatest
        self.uniform = uniform
        self.pol_weights = np.array([0.1, 0.2, 0.3, 0.4])

        self.buffer = Buffer.GAEVBuffer(
            env, horizon=horizon, nlatest=nlatest, gamma=gamma, lam=lam,
            compute_neglogp_pik=self.compute_neglogp_pik, compute_v_pik=self.compute_v_pik)
        self._build_network()
        self._build_train_op()
        super().__init__()

    def _build_network(self):
        self.actor = ActorMLP(self.ac_shape, name='pi')
        # self.actor_pik = ActorMLP(self.ac_shape, name='pik')
        self.critic = CriticMLP(name='vf')

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
        self.neglogp_pik_ph = neglogp_pik_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="neglogp_pik_ph")
        self.weights_ph = weights_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="weights_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")

        # Probability distribution
        logstd1 = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.ac_shape[0]),
            initializer=tf.zeros_initializer)

        # Interative with env
        mu1 = self.actor(ob1_ph)
        dist1 = DiagGaussianPd(mu1, logstd1)
        pi1 = dist1.sample()
        neglogp1 = tf.reduce_sum(dist1.neglogp(pi1), axis=1)

        v1 = self.critic(ob1_ph)

        get_action_ops = [mu1, pi1, v1, neglogp1]

        # Train batch data
        mu = self.actor(obs_ph)
        logstd = tf.tile(logstd1, (tf.shape(mu)[0], 1))
        dist = DiagGaussianPd(mu, logstd)
        neglopac = tf.reduce_sum(dist.neglogp(ac_ph), axis=1)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        v = self.critic(obs_ph)

        if self.vf_clip:
            valclipped = val_ph + \
                tf.clip_by_value(v - val_ph, -self.clip_ratio, self.clip_ratio)
            vf_loss1 = tf.square(v - ret_ph)
            vf_loss2 = tf.square(valclipped - ret_ph)
            vf_loss = 0.5 * tf.reduce_mean(weights_ph * tf.maximum(vf_loss1, vf_loss2))
        else:
            vf_loss = 0.5 * tf.reduce_mean(weights_ph * tf.square(v - ret_ph))

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio = tf.exp(neglogp_old_ph - neglopac)
        ratio_pik = tf.exp(neglogp_old_ph - neglogp_pik_ph)
        pi_loss1 = -adv_ph * ratio
        pi_loss2 = -adv_ph * tf.clip_by_value(
            ratio, ratio_pik - self.clip_ratio, ratio_pik + self.clip_ratio)
        pi_loss = tf.reduce_mean(weights_ph * tf.maximum(pi_loss1, pi_loss2))

        # Total loss
        loss = pi_loss - meanent * self.ent_coef + vf_loss * self.vf_coef

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        tv = 0.5 * tf.reduce_mean(weights_ph * tf.abs(ratio - ratio_pik))
        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogp_pik_ph - neglopac))
        absratio = tf.reduce_mean(tf.abs(ratio - 1.0) + 1.0)
        ratioclipped = tf.where(
            adv_ph > 0,
            ratio > (ratio_pik + self.clip_ratio),
            ratio < (ratio_pik - self.clip_ratio))
        ratioclipfrac = tf.reduce_mean(tf.cast(ratioclipped, tf.float32))

        # Optimizers
        pi_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr_ph)
        vf_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.lr)
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        grads_and_vars = pi_optimizer.compute_gradients(
            pi_loss, var_list=pi_params)
        grads, vars = zip(*grads_and_vars)
        if self.grad_clip:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
        gradclipped = _grad_norm > self.max_grad_norm if self.grad_clip else tf.zeros(())
        grads_and_vars = list(zip(grads, vars))

        pi_train_op = pi_optimizer.apply_gradients(grads_and_vars)
        vf_train_op = vf_optimizer.minimize(vf_loss, var_list=vf_params)

        self.get_action_ops = get_action_ops
        self.neglopac = neglopac
        self.v1 = v1
        self.v = v
        self.absratio = absratio
        self.pi_loss = pi_loss
        self.vf_loss = vf_loss
        self.approxkl = approxkl
        self.meanent = meanent
        self.tv = tv
        self.ratioclipfrac = ratioclipfrac
        self.gradclipped = gradclipped
        # self.train_op = train_op
        self.pi_train_op = pi_train_op
        self.vf_train_op = vf_train_op

        self.losses = [pi_loss, vf_loss, meanent, approxkl]
        self.infos = [absratio, ratioclipfrac, gradclipped, tv]

    def update(self, frac, logger):
        buf_data = self.buffer.vtrace()
        [obs_all, ac_all, adv_all, ret_all, v_all, neglogp_old_all, neglogp_pik_all] = buf_data
        rho_all = np.exp(neglogp_old_all - neglogp_pik_all)

        if self.uniform:
            weights_all = np.ones(obs_all.shape[0])
        else:
            M_active = obs_all.shape[0] // self.horizon
            weights_active = self.pol_weights[-M_active:]
            weights_active = weights_active / np.sum(weights_active)
            weights_active *= M_active
            weights_all = np.repeat(weights_active, self.horizon)

        # lr = self.lr if self.fixed_lr else self.lr * frac

        pi_loss_buf = []
        vf_loss_buf = []
        ent_buf = []
        kl_buf = []
        tv_buf = []
        ratio_buf = []
        ratioclipfrac_buf = []
        gradclipped_buf = []

        length = obs_all.shape[0]
        minibatch = length // self.nminibatches
        indices = np.arange(length)
        for _ in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, length, minibatch):
                end = start + minibatch
                mbinds = indices[start:end]
                advs = adv_all[mbinds]
                rhos = rho_all[mbinds]
                advs_mean = np.mean(advs * rhos) / np.mean(rhos)
                advs_std = np.std(advs * rhos)
                advs_norm = (advs - advs_mean) / (advs_std + 1e-8)
                inputs = {
                    self.obs_ph: obs_all[mbinds],
                    self.ac_ph: ac_all[mbinds],
                    self.adv_ph: advs_norm,
                    self.ret_ph: ret_all[mbinds],
                    self.val_ph: v_all[mbinds],
                    self.neglogp_old_ph: neglogp_old_all[mbinds],
                    self.neglogp_pik_ph: neglogp_pik_all[mbinds],
                    self.weights_ph: weights_all[mbinds],
                    self.lr_ph: self.lr
                }

                infos, losses, _, _ = self.sess.run(
                    [self.infos, self.losses, self.pi_train_op, self.vf_train_op],
                    feed_dict=inputs)
                # Unpack losses
                pi_loss, vf_loss, ent, kl = losses
                pi_loss_buf.append(pi_loss)
                vf_loss_buf.append(vf_loss)
                ent_buf.append(ent)
                kl_buf.append(kl)
                # Unpack infos
                ratio, ratioclipfrac, gradclipped, tv = infos
                ratio_buf.append(ratio)
                ratioclipfrac_buf.append(ratioclipfrac)
                gradclipped_buf.append(gradclipped)
                tv_buf.append(tv)

            tv_inputs = {
                self.obs_ph: obs_all,
                self.ac_ph: ac_all,
                self.neglogp_old_ph: neglogp_old_all,
                self.neglogp_pik_ph: neglogp_pik_all,
                self.weights_ph: weights_all
            }
            tv_all = self.sess.run(self.tv, feed_dict=tv_inputs)

        if tv_all > self.thresh * 0.5 * self.clip_ratio * 2:
            self.lr /= (1 + self.alpha)
        elif tv_all < self.thresh * 0.5 * self.clip_ratio:
            self.lr *= (1 + self.alpha)

        # Here you can add any information you want to log!
        logger.logkv("loss/gradclipfrac", np.mean(gradclipped))
        logger.logkv("loss/lr", self.lr)
        logger.logkv("loss/ratio", np.mean(ratio_buf))
        logger.logkv("loss/ratioclipfrac", np.mean(ratioclipfrac_buf))
        logger.logkv("loss/tv", tv_all)

        self.buffer.update()

        return [np.mean(pi_loss_buf), np.mean(vf_loss_buf),
                np.mean(ent_buf), np.mean(kl_buf)]

    def select_action(self, obs, deterministic=False):
        [mu, pi, v, neglogp] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        ac = mu if deterministic else pi
        return pi, v, neglogp

    def compute_v(self, obs):
        v = self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        return v

    def compute_v_pik(self, obs):
        return self.sess.run(self.v, feed_dict={self.obs_ph: obs})

    def compute_neglogp_pik(self, obs, ac):
        inputs = {
            self.obs_ph: obs,
            self.ac_ph: ac
        }
        return self.sess.run(self.neglopac, feed_dict=inputs)
