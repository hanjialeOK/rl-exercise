import numpy as np
import tensorflow as tf
import os

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
                 clip_ratio=0.4, lr=3e-4, train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, jtarg=1e-3,
                 horizon=2048, nminibatches=32, gamma=0.99, lam=0.95,
                 alpha=1, grad_clip=False, vf_clip=True, fixed_lr=False,
                 nlatest=64):
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
        self.jtarg = jtarg
        self.alpha = alpha
        self.grad_clip = grad_clip
        self.vf_clip = vf_clip
        self.fixed_lr = fixed_lr

        self.buffer = Buffer.DISCBuffer(
            env, horizon=horizon, nlatest=nlatest, gamma=gamma, lam=lam,
            compute_v_pik=self.compute_v_pik, compute_neglogp_pik=self.compute_neglogp_pik)
        self._build_network()
        self._build_train_op()
        self.sync_op = self._build_sync_op()
        super().__init__()

    def _build_network(self):
        self.actor = ActorMLP(self.ac_shape, name='pi')
        self.actor_pik = ActorMLP(self.ac_shape, name='pik')
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
        self.val_ph = val_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="val_ph")
        self.neglogp_dw_old_ph = neglogp_dw_old_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.ac_shape, dtype=tf.float32, name="neglogp_dw_old_ph")
        self.neglogp_dw_pik_ph = neglogp_dw_pik_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.ac_shape, dtype=tf.float32, name="neglogp_dw_pik_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")
        self.alpha_ph = alpha_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="alpha_ph")
        self.on_policy_ph = on_policy_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="on_policy_ph")

        # Probability distribution
        logstd1 = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.ac_shape[0]),
            initializer=tf.zeros_initializer)
        logstd1_pik = tf.compat.v1.get_variable(
            name='pik/logstd_pik', shape=(1, self.ac_shape[0]),
            initializer=tf.zeros_initializer)

        # Interative with env
        mu1 = self.actor(ob1_ph)
        dist1 = DiagGaussianPd(mu1, logstd1)
        pi1 = dist1.sample()
        neglogp1_dw = dist1.neglogp(pi1)

        v1 = self.critic(ob1_ph)

        get_action_ops = [mu1, pi1, v1, neglogp1_dw]

        # Train batch data
        mu = self.actor(obs_ph)
        logstd = tf.tile(logstd1, (tf.shape(mu)[0], 1))
        dist = DiagGaussianPd(mu, logstd)
        neglogpac_dw = dist.neglogp(ac_ph)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        mu_pik = tf.stop_gradient(self.actor_pik(obs_ph))
        logstd_pik = tf.tile(logstd1_pik, (tf.shape(mu_pik)[0], 1))
        dist_pik = DiagGaussianPd(mu_pik, logstd_pik)
        kl = tf.reduce_sum(dist.kl(dist_pik), axis=1)
        meankl = tf.reduce_mean(kl)

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
        neglogp_old = tf.reduce_sum(neglogp_dw_old_ph, axis=1)
        neglogp_pik = tf.reduce_sum(neglogp_dw_pik_ph, axis=1)
        neglogpac = tf.reduce_sum(neglogpac_dw, axis=1)
        ratio_dw = tf.exp(neglogp_dw_old_ph - neglogpac_dw)
        ratio_dw_pik = tf.exp(neglogp_dw_old_ph - neglogp_dw_pik_ph)
        ratio_dw_clip = tf.clip_by_value(
            ratio_dw, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        # ratio_dw_min = tf.where(
        #     adv_ph > 0,
        #     tf.minimum(ratio_dw, ratio_dw_clip),
        #     tf.maximum(ratio_dw, ratio_dw_clip))
        ratio = tf.reduce_prod(ratio_dw, axis=1)
        ratio_pik = tf.reduce_prod(ratio_dw_pik, axis=1)

        sgn = tf.expand_dims(tf.sign(adv_ph), axis=1)
        ratio_min = tf.reduce_prod(sgn * tf.minimum(ratio_dw * sgn, ratio_dw_clip * sgn), axis=1)
        # pi_loss = -tf.reduce_mean(adv_ph * ratio_min )
        pi_loss = -tf.reduce_mean(adv_ph * ratio_min / tf.stop_gradient(tf.reduce_mean(ratio_min)))

        pi_loss_ctl = 0.5 * tf.reduce_mean(tf.square(neglogp_old - neglogpac)*on_policy_ph)
        # pi_loss_ctl = tf.reduce_mean(kl*on_policy_ph)
        pi_loss += alpha_ph * pi_loss_ctl

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogp_pik - neglogpac))
        absratio = tf.reduce_mean(tf.abs(ratio - 1.0) + 1.0)
        ratioclipped = tf.where(
            adv_ph > 0,
            ratio_dw > (1.0 + self.clip_ratio),
            ratio_dw < (1.0 - self.clip_ratio))
        ratioclipfrac = tf.reduce_mean(tf.cast(ratioclipped, tf.float32))
        tv_on = 0.5 * tf.reduce_mean(tf.abs(ratio - ratio_pik)*on_policy_ph)
        tv = 0.5 * tf.reduce_mean(tf.abs(ratio - ratio_pik))

        # Total loss
        loss = pi_loss - meanent * self.ent_coef + vf_loss * self.vf_coef

        # Optimizers
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr_ph, epsilon=1e-5)
        params = self._get_var_list('pi') + self._get_var_list('vf')
        grads_and_vars = optimizer.compute_gradients(loss, var_list=params)
        grads, vars = zip(*grads_and_vars)
        if self.grad_clip:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, self.max_grad_norm)
        gradclipped = _grad_norm > self.max_grad_norm if self.grad_clip else tf.zeros(())
        grads_and_vars = list(zip(grads, vars))

        train_op = optimizer.apply_gradients(grads_and_vars)

        self.get_action_ops = get_action_ops
        self.v1 = v1
        self.v = v
        self.neglogpac_dw = neglogpac_dw
        self.absratio = absratio
        self.pi_loss_ctl = pi_loss_ctl
        self.pi_loss = pi_loss
        self.vf_loss = vf_loss
        self.meankl = meankl
        self.meanent = meanent
        self.ratioclipfrac = ratioclipfrac
        self.gradclipped = gradclipped
        self.train_op = train_op

        self.losses = [pi_loss_ctl, pi_loss, vf_loss, meanent, meankl]
        self.infos = [absratio, ratioclipfrac, gradclipped, tv_on, tv]

    def _build_sync_op(self):
        sync_qt_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_old = self._get_var_list('pik')

        for (newv, oldv) in zip(pi_params, pi_params_old):
            sync_qt_ops.append(oldv.assign(newv, use_locking=True))
        return sync_qt_ops

    def update(self, frac, logger):
        buf_data = self.buffer.vtrace()
        [obs_all, ac_all, adv_all, ret_all, v_all, neglogp_dw_old_all, neglogp_dw_pik_all] = buf_data
        neglogp_pik_all = np.sum(neglogp_dw_pik_all, axis=1)
        neglogp_old_all = np.sum(neglogp_dw_old_all, axis=1)
        rho_all = np.exp(neglogp_pik_all - neglogp_old_all)
        rho_dw_all = np.exp(neglogp_dw_old_all - neglogp_dw_pik_all)

        absrho_dw_all = np.abs(rho_dw_all - 1) + 1
        n_oldtrajs = rho_dw_all.shape[0] // self.horizon - 1

        filter_inds = np.array([], dtype=np.int64)
        # Filter old trajs
        for s in range(n_oldtrajs):
            start = s*self.horizon
            end = (s+1)*self.horizon
            if np.mean(absrho_dw_all[start:end]) <= 1 + 0.1:
                filter_inds = np.concatenate([filter_inds, np.arange(start, end)])
        # Add the latest traj
        piktraj_inds = np.arange(obs_all.shape[0])[-self.horizon:]
        filter_inds = np.concatenate([filter_inds, piktraj_inds])

        obs_filter = obs_all[filter_inds]
        ac_filter = ac_all[filter_inds]
        adv_filter = adv_all[filter_inds]
        ret_filter = ret_all[filter_inds]
        neglogp_dw_old_filter = neglogp_dw_old_all[filter_inds]
        v_filter = v_all[filter_inds]
        neglogp_dw_pik_filter = neglogp_dw_pik_all[filter_inds]
        rho_filter = rho_all[filter_inds]

        n_trajs_active = obs_filter.shape[0] // self.horizon

        on_policy = np.zeros(obs_filter.shape[0])
        on_policy[-self.horizon:] = n_trajs_active

        lr = self.lr if self.fixed_lr else np.maximum(self.lr * frac, 1e-4)

        pi_loss_ctl_buf = []
        pi_loss_buf = []
        vf_loss_buf = []
        ent_buf = []
        kl_buf = []
        ratio_buf = []
        ratioclipfrac_buf = []
        gradclipped_buf = []
        tv_on_buf = []
        tv_buf = []

        self.sess.run(self.sync_op)

        active_length = obs_filter.shape[0]
        indices = np.arange(active_length)
        minibatch_off = (active_length - self.horizon) // self.nminibatches
        minibatch_on = self.horizon // self.nminibatches
        for _ in range(self.train_iters):
            # Randomize the indexes
            # np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            # for start in range(0, length, minibatch):
            for _ in range(self.nminibatches):
                if minibatch_off > 0:
                    idx_off = np.random.choice(indices[:-self.horizon], minibatch_off)
                else:
                    idx_off = []
                idx_on = np.random.choice(indices[-self.horizon:], minibatch_on)
                mbinds = np.concatenate([idx_off, idx_on]).astype(np.int64)
                advs = adv_filter[mbinds]
                rhos = rho_filter[mbinds]
                rhos = np.minimum(rhos, 1.0)
                advs_mean = np.mean(advs * rhos) / np.mean(rhos)
                advs_std = np.std(advs * rhos)
                advs_norm = (advs - advs_mean) / (advs_std + 1e-8)
                inputs = {
                    self.obs_ph: obs_filter[mbinds],
                    self.ac_ph: ac_filter[mbinds],
                    self.adv_ph: advs_norm,
                    self.ret_ph: ret_filter[mbinds],
                    self.val_ph: v_filter[mbinds],
                    self.neglogp_dw_old_ph: neglogp_dw_old_filter[mbinds],
                    self.neglogp_dw_pik_ph: neglogp_dw_pik_filter[mbinds],
                    self.lr_ph: lr,
                    self.alpha_ph: self.alpha,
                    self.on_policy_ph: on_policy[mbinds]
                }

                infos, losses, _ = self.sess.run(
                    [self.infos, self.losses, self.train_op],
                    feed_dict=inputs)
                # Unpack losses
                pi_loss_ctl, pi_loss, vf_loss, ent, kl = losses
                pi_loss_ctl_buf.append(pi_loss_ctl)
                pi_loss_buf.append(pi_loss)
                vf_loss_buf.append(vf_loss)
                ent_buf.append(ent)
                kl_buf.append(kl)
                # Unpack infos
                ratio, ratioclipfrac, gradclipped, tv_on, tv = infos
                ratio_buf.append(ratio)
                ratioclipfrac_buf.append(ratioclipfrac)
                gradclipped_buf.append(gradclipped)
                tv_on_buf.append(tv_on)
                tv_buf.append(tv)

            # tv_inputs = {
            #     self.obs_ph: obs_filter,
            #     self.ac_ph: ac_filter,
            #     self.neglogp_dw_old_ph: neglogp_dw_old_filter,
            #     self.neglogp_dw_pik_ph: neglogp_dw_pik_filter,
            #     self.on_policy_ph: on_policy
            # }
            # pi_loss_ctl_all = self.sess.run(self.pi_loss_ctl, feed_dict=tv_inputs)

        # if tv_all > self.thresh * 0.5 * self.clip_ratio * 2:
        #     self.lr /= (1 + self.alpha)
        # elif tv_all < self.thresh * 0.5 * self.clip_ratio:
        #     self.lr *= (1 + self.alpha)
        pi_loss_ctl_mean = np.mean(pi_loss_ctl_buf)

        if pi_loss_ctl_mean > self.jtarg * 1.5:
            self.alpha *= 2
        elif pi_loss_ctl_mean < self.jtarg / 1.5:
            self.alpha /= 2
        self.alpha = np.clip(self.alpha, 2**(-10), 64)

        logger.logkv("loss/gradclipfrac", np.mean(gradclipped))
        logger.logkv("loss/lr", lr)
        logger.logkv("loss/ratio", np.mean(ratio_buf))
        logger.logkv("loss/ratioclipfrac", np.mean(ratioclipfrac_buf))
        logger.logkv("loss/pi_loss_ctl", pi_loss_ctl_mean)
        logger.logkv("loss/trajs_active", n_trajs_active)
        logger.logkv("loss/tv", np.mean(tv_buf))
        logger.logkv("loss/tv_on", np.mean(tv_on_buf))
        logger.logkv("loss/alpha", self.alpha)

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
        return self.sess.run(self.neglogpac_dw, feed_dict=inputs)
