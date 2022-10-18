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
                 clip_ratio=0.4, lr=3e-4, train_iters=10, target_kl=1e-3,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=2048, nminibatches=32, gamma=0.99, lam=0.95,
                 grad_clip=False, vf_clip=True, fixed_lr=False, beta=1,
                 thresh=0.4, alpha=0.03, nlatest=64, uniform=True,
                 geppo=True, clip_ratio2=0.8):
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
        self.beta = beta
        self.geppo = geppo
        self.clip_ratio2 = clip_ratio2

        self.buffer = Buffer.GAEVBuffer(
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
        self.neglogp_old_ph = neglogp_old_ph = tf.compat.v1.placeholder(
            shape=(None, ), dtype=tf.float32, name="neglogp_old_ph")
        self.neglogp_pik_ph = neglogp_pik_ph = tf.compat.v1.placeholder(
            shape=(None, ), dtype=tf.float32, name="neglogp_pik_ph")
        self.weights_ph = weights_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="weights_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")
        self.beta_ph = beta_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="beta_ph")
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
        neglogp1 = tf.reduce_sum(dist1.neglogp(pi1), axis=1)

        v1 = self.critic(ob1_ph)

        get_action_ops = [mu1, logstd1, pi1, v1, neglogp1]

        # Train batch data
        mu = self.actor(obs_ph)
        logstd = tf.tile(logstd1, (tf.shape(mu)[0], 1))
        dist = DiagGaussianPd(mu, logstd)
        neglogpac = tf.reduce_sum(dist.neglogp(ac_ph), axis=1)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        mu_pik = tf.stop_gradient(self.actor_pik(obs_ph))
        logstd_pik = tf.tile(logstd1_pik, (tf.shape(mu_pik)[0], 1))
        dist_pik = DiagGaussianPd(mu_pik, logstd_pik)
        kl = tf.reduce_sum(dist_pik.kl(dist), axis=1)
        meankl = tf.reduce_mean(kl)

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
        ratio = tf.exp(neglogp_old_ph - neglogpac)
        ratio_pik = tf.exp(neglogp_old_ph - neglogp_pik_ph)
        center = ratio_pik if self.geppo else 1.0
        ratio_clip = tf.clip_by_value(
            ratio, center - self.clip_ratio, center + self.clip_ratio)
        ratio_clip = tf.clip_by_value(
            ratio_clip, 1.0 - self.clip_ratio2, 1.0 + self.clip_ratio2)
        pi_loss1 = -adv_ph * ratio
        pi_loss2 = -adv_ph * ratio_clip
        pi_loss = tf.reduce_mean(weights_ph * tf.maximum(pi_loss1, pi_loss2))

        # pi_loss_ctl = 0.5 * tf.reduce_mean(kl*on_policy_ph)
        pi_loss_ctl = 0.5 * tf.reduce_mean(tf.square(neglogp_old_ph - neglogpac)*on_policy_ph)
        # pi_loss_ctl = 0.5 * tf.reduce_mean((ratio - 1.0 - tf.log(ratio))*on_policy_ph)
        pi_loss += beta_ph * pi_loss_ctl

        # Total loss
        loss = pi_loss - meanent * self.ent_coef + vf_loss * self.vf_coef

        # Info (useful to watch during learning)
        approxkl = 0.5 * tf.reduce_mean(tf.square(neglogp_pik_ph - neglogpac))
        absratio = tf.reduce_mean(tf.abs(ratio - 1.0) + 1.0)
        max_ratio = tf.reduce_max(ratio)
        min_ratio = tf.reduce_min(ratio)
        ratioclipped1 = tf.where(
            adv_ph > 0,
            ratio > center + self.clip_ratio,
            ratio < center - self.clip_ratio)
        ratioclipped2 = tf.where(
            adv_ph > 0,
            ratio > 1.0 + self.clip_ratio2,
            ratio < 1.0 - self.clip_ratio2)
        ratioclipfrac1 = tf.reduce_mean(tf.cast(ratioclipped1, tf.float32))
        ratioclipfrac2 = tf.reduce_mean(tf.cast(ratioclipped2, tf.float32))
        ratioclipfrac = tf.reduce_mean(tf.cast(tf.logical_or(ratioclipped1, ratioclipped2), tf.float32))
        tv_on = 0.5 * tf.reduce_mean(weights_ph * tf.abs(ratio - ratio_pik)*on_policy_ph)
        tv = 0.5 * tf.reduce_mean(weights_ph * tf.abs(ratio - ratio_pik))

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
        self.neglogpac = neglogpac
        self.v1 = v1
        self.v = v
        self.train_op = train_op

        self.stats_list = [pi_loss_ctl, pi_loss, vf_loss, meanent, meankl, approxkl, max_ratio, min_ratio,
                           absratio, ratioclipfrac1, ratioclipfrac2, ratioclipfrac, gradclipped, tv_on, tv]
        self.loss_names = ['pi_loss_ctl', 'pi_loss', 'vf_loss', 'entropy', 'kl', 'approxkl', 'max_ratio', 'min_ratio',
                           'absratio', 'ratioclipfrac1', 'ratioclipfrac2', 'ratioclipfrac', 'gradclipped', 'tv_on', 'tv']
        assert len(self.stats_list) == len(self.loss_names)

    def _build_sync_op(self):
        sync_qt_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_old = self._get_var_list('pik')

        for (newv, oldv) in zip(pi_params, pi_params_old):
            sync_qt_ops.append(oldv.assign(newv, use_locking=True))
        return sync_qt_ops

    def update(self, frac, logger):
        buf_data = self.buffer.vtrace()
        [obs_all, ac_all, adv_all, ret_all, v_all, neglogp_old_all, neglogp_pik_all] = buf_data
        weights_all = np.ones(obs_all.shape[0])
        rho_all = np.exp(neglogp_old_all - neglogp_pik_all)

        absrho_all = np.abs(rho_all - 1) + 1
        n_oldtrajs = rho_all.shape[0] // self.horizon - 1

        filter_inds = np.array([], dtype=np.int64)
        thresh = self.thresh
        # thresh = np.maximum(self.thresh * frac, 0.2)
        # Filter old trajs
        for s in range(n_oldtrajs):
            start = s*self.horizon
            end = (s+1)*self.horizon
            if np.mean(absrho_all[start:end]) <= 1 + thresh:
                filter_inds = np.concatenate([filter_inds, np.arange(start, end)])
        # Add the latest traj
        newtraj_inds = np.arange(obs_all.shape[0])[-self.horizon:]
        filter_inds = np.concatenate([filter_inds, newtraj_inds])

        obs_filted = obs_all[filter_inds]
        ac_filted = ac_all[filter_inds]
        adv_filted = adv_all[filter_inds]
        ret_filted = ret_all[filter_inds]
        neglogp_old_filted = neglogp_old_all[filter_inds]
        v_filted = v_all[filter_inds]
        neglogp_pik_filted = neglogp_pik_all[filter_inds]
        rho_filted = rho_all[filter_inds]
        weights_filted = weights_all[filter_inds]

        n_trajs_active = obs_filted.shape[0] // self.horizon

        on_policy = np.zeros(obs_filted.shape[0])
        on_policy[-self.horizon:] = n_trajs_active

        lr = self.lr if self.fixed_lr else np.maximum(self.lr * frac, 1e-4)
        # lr = self.lr
        # self.thresh = np.maximum(self.thresh * frac, 0.1)

        self.sess.run(self.sync_op)

        active_length = obs_filted.shape[0]
        indices = np.arange(active_length)
        minibatch_off = (active_length - self.horizon) // self.nminibatches
        minibatch_on = self.horizon // self.nminibatches

        mblossvals = []
        for _ in range(self.train_iters):
            # Randomize the indexes
            # np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for _ in range(self.nminibatches):
                if minibatch_off > 0:
                    idx_off = np.random.choice(indices[:-self.horizon], minibatch_off)
                else:
                    idx_off = []
                idx_on = np.random.choice(indices[-self.horizon:], minibatch_on)
                mbinds = np.concatenate([idx_off, idx_on]).astype(np.int64)
                advs = adv_filted[mbinds]
                rhos = rho_filted[mbinds]
                rhos = np.minimum(rhos, 1.0)
                advs_mean = np.mean(advs * rhos) / np.mean(rhos)
                advs_std = np.std(advs * rhos)
                advs_norm = (advs - advs_mean) / (advs_std + 1e-8)
                inputs = {
                    self.obs_ph: obs_filted[mbinds],
                    self.ac_ph: ac_filted[mbinds],
                    self.adv_ph: advs_norm,
                    self.ret_ph: ret_filted[mbinds],
                    self.val_ph: v_filted[mbinds],
                    self.neglogp_old_ph: neglogp_old_filted[mbinds],
                    self.neglogp_pik_ph: neglogp_pik_filted[mbinds],
                    self.lr_ph: lr,
                    self.beta_ph: self.beta,
                    self.on_policy_ph: on_policy[mbinds],
                    self.weights_ph: weights_filted[mbinds]
                }

                losses = self.sess.run(self.stats_list + [self.train_op], feed_dict=inputs)[:-1]
                mblossvals.append(losses)

            # tv_inputs = {
            #     self.obs_ph: obs_filted,
            #     self.ac_ph: ac_filted,
            #     self.neglogp_old_ph: neglogp_old_filted,
            #     self.neglogp_pik_ph: neglogp_pik_filted,
            #     self.on_policy_ph: on_policy,
            #     self.weights_ph: weights_filted
            # }
            # tv_all, pi_loss_ctl_all = self.sess.run([self.tv, self.pi_loss_ctl], feed_dict=tv_inputs)

        # if tv_all > 0.5 * self.clip_ratio:
        #     self.lr /= (1 + self.alpha)
        # elif tv_all < 0.5 * self.clip_ratio * 0.5:
        #     self.lr *= (1 + self.alpha)
        lossvals = np.mean(mblossvals, axis=0)
        pi_loss_ctl_mean = lossvals[0]

        if pi_loss_ctl_mean > self.target_kl * 1.5:
            self.beta *= 2
        elif pi_loss_ctl_mean < self.target_kl / 1.5:
            self.beta /= 2
        self.beta = np.clip(self.beta, 2**(-10), 64)

        # Here you can add any information you want to log!
        for (lossval, lossname) in zip(lossvals, self.loss_names):
            logger.logkv('loss/' + lossname, lossval)
        logger.logkv("loss/lr", lr)
        logger.logkv("loss/trajs_active", n_trajs_active)
        logger.logkv("loss/beta", self.beta)
        logger.logkv("loss/thresh", thresh)
        logger.logkv("loss/clip1", self.clip_ratio)
        logger.logkv("loss/clip2", self.clip_ratio2)

        self.buffer.update()

    def select_action(self, obs, deterministic=False):
        [mu, logstd, pi, v, neglogp] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        ac = mu if deterministic else pi
        return pi, v, neglogp, mu, logstd

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
        return self.sess.run(self.neglogpac, feed_dict=inputs)
