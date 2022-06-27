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


class PPOAgent(BaseAgent):
    def __init__(self, sess, summary_writer, obs_dim, act_dim,
                 clip_ratio=0.4, lr=3e-4, train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_j=1e-3,
                 horizon=2048, nminibatches=32, gamma=0.99, lam=0.95,
                 alpha=1, grad_clip=False, vf_clip=True, fixed_lr=False,
                 nlatest=64, uniform=False, gedisc=False):
        self.sess = sess
        self.summary_writer = summary_writer
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.horizon = horizon
        self.nminibatches = nminibatches
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_j = target_j
        self.alpha = alpha
        self.grad_clip = grad_clip
        self.vf_clip = vf_clip
        self.fixed_lr = fixed_lr
        self.gedisc = gedisc

        self.buffer = Buffer.DISCBuffer(
            obs_dim, act_dim, size=horizon, nlatest=nlatest, gamma=gamma, lam=lam, uniform=uniform)
        self._build_network()
        self._build_train_op()
        self.saver = self._build_saver()
        self.sync_op = self._build_sync_op()

    def _build_network(self):
        self.actor = ActorMLP(self.act_dim, name='pi')
        self.actor_pik = ActorMLP(self.act_dim, name='pik')
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
        self.logp_disc_old_ph = logp_disc_old_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.act_dim, dtype=tf.float32, name="logp_disc_old_ph")
        self.logp_disc_pik_ph = logp_disc_pik_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.act_dim, dtype=tf.float32, name="logp_disc_pik_ph")
        self.val_ph = val_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="val_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")
        self.alpha_ph = alpha_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="alpha_ph")
        self.on_policy_ph = on_policy_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="on_policy_ph")

        # Probability distribution
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.act_dim[0]),
            initializer=tf.zeros_initializer)
        logstd_pik = tf.compat.v1.get_variable(
            name='pik/logstd_pik', shape=(1, self.act_dim[0]),
            initializer=tf.zeros_initializer)
        std = tf.exp(logstd)

        # Interative with env
        mu1 = self.actor(ob1_ph)
        dist1 = tfp.distributions.Normal(loc=mu1, scale=std)
        pi1 = dist1.sample()
        # logp_pi1 = tf.reduce_sum(dist1.log_prob(pi1), axis=1)
        logp_pi1_disc = dist1.log_prob(pi1)

        v1 = self.critic(ob1_ph)

        get_action_ops = [mu1, pi1, v1, logp_pi1_disc]

        # Train batch data
        mu = self.actor(obs_ph)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        # logp_a = tf.reduce_sum(dist.log_prob(act_ph), axis=1)
        logp_a_disc = dist.log_prob(act_ph)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        mu_pik = tf.stop_gradient(self.actor_pik(obs_ph))
        std_pik = tf.stop_gradient(tf.exp(logstd_pik))
        dist_pik = tfp.distributions.Normal(loc=mu_pik, scale=std_pik)
        kl = tf.reduce_sum(dist.kl_divergence(dist_pik), axis=1)
        meankl = tf.reduce_mean(kl)
        # approxkl = 0.5 * tf.reduce_mean(tf.square(logp_pik_ph - logp_a))

        v = self.critic(obs_ph)

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio_disc = tf.exp(logp_a_disc - logp_disc_old_ph)
        ratio_disc_pik = tf.exp(logp_a_disc - logp_disc_pik_ph)
        center = ratio_disc_pik if self.gedisc else 1.0
        ratio_disc_clip = tf.clip_by_value(
            ratio_disc, center - self.clip_ratio, center + self.clip_ratio)
        ratio_disc_min = tf.where(
            adv_ph > 0,
            tf.minimum(ratio_disc, ratio_disc_clip),
            tf.maximum(ratio_disc, ratio_disc_clip))
        ratio = tf.reduce_prod(ratio_disc, axis=1)
        ratio_min = tf.reduce_prod(ratio_disc_min, axis=1)

        # sign_disc = tf.ones_like(ratio_disc) * tf.expand_dims(tf.sign(adv_ph), 1)
        # r = tf.reduce_prod(sign_disc * tf.minimum(ratio_disc * sign_disc, ratio_disc_clip * sign_disc), axis=1)

        # pi_loss1 = -adv_ph * ratio
        # pi_loss2 = -adv_ph * tf.clip_by_value(
        #     ratio2, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        # pi_loss = tf.reduce_mean(tf.maximum(pi_loss1, pi_loss2))
        logp_pik = tf.reduce_sum(logp_disc_pik_ph, axis=1)
        logp_a = tf.reduce_sum(logp_a_disc, axis=1)
        approxkl = 0.5 * tf.reduce_mean(tf.square(logp_pik - logp_a))
        pi_loss_ctl = 0.5 * tf.reduce_mean(tf.square(logp_pik - logp_a)*on_policy_ph)
        # pi_loss_ctl = tf.reduce_mean(kl*on_policy_ph)

        pi_loss = -tf.reduce_mean(adv_ph * ratio_min / tf.stop_gradient(tf.reduce_mean(ratio_min)))
        # pi_loss = -tf.reduce_mean(adv_ph * r / tf.stop_gradient(tf.reduce_mean(r)))
        pi_loss += alpha_ph * pi_loss_ctl

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
        # meankl = 0.5 * tf.reduce_mean(tf.square(logp_old - logp_a))
        absratio = tf.reduce_mean(tf.abs(ratio - 1.0) + 1.0)
        ratioclipped = tf.where(
            adv_ph > 0,
            ratio_disc > (1.0 + self.clip_ratio),
            ratio_disc < (1.0 - self.clip_ratio))
        ratioclipfrac = tf.reduce_mean(tf.cast(ratioclipped, tf.float32))

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
        self.logp_a_disc = logp_a_disc
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
        self.infos = [absratio, ratioclipfrac, gradclipped]

    def _build_sync_op(self):
        sync_qt_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_old = self._get_var_list('pik')

        for (newv, oldv) in zip(pi_params, pi_params_old):
            sync_qt_ops.append(oldv.assign(newv, use_locking=True))
        return sync_qt_ops

    def update(self, frac, log2board, step):
        # obs_all, obs2_all, ac_all = self.buffer.get_obs()
        # v_pik = self.sess.run(self.v, feed_dict={ self.obs_ph: obs_all })
        # v2_pik = self.sess.run(self.v, feed_dict={ self.obs_ph: obs2_all })
        # logp_inputs = {self.obs_ph: obs_all, self.act_ph: ac_all}
        # logp_pik = self.sess.run(self.logp_a, feed_dict=logp_inputs)
        buf_data = self.buffer.vtrace(self.compute_v_pik, self.compute_logp_pik)
        self.buffer.update()
        [obs_all, ac_all, adv_all, ret_all, logp_disc_old_all, v_all, logp_disc_pik_all, weights_all] = buf_data
        logp_pik_all = np.sum(logp_disc_pik_all, axis=1)
        logp_old_all = np.sum(logp_disc_old_all, axis=1)
        rho_all = np.exp(logp_pik_all - logp_old_all)
        rho_disc_all = np.exp(logp_disc_pik_all - logp_disc_old_all)

        absrho_all = np.abs(rho_disc_all - 1) + 1
        n_trajs = rho_disc_all.shape[0] // self.horizon
        meanabsrho_buf = np.zeros(n_trajs)

        obs_filter = []
        ac_filter = []
        adv_filter = []
        ret_filter = []
        logp_disc_old_filter = []
        v_filter = []
        logp_disc_pik_filter = []
        weights_filter = []
        rho_filter = []

        self.sess.run(self.sync_op)

        for s in range(n_trajs):
            start = s*self.horizon
            end = (s+1)*self.horizon
            meanabsrho_buf[s] = np.mean(absrho_all[start:end])
            if meanabsrho_buf[s] <= 1 + 0.1:
                obs_filter.append(obs_all[start:end])
                ac_filter.append(ac_all[start:end])
                adv_filter.append(adv_all[start:end])
                ret_filter.append(ret_all[start:end])
                logp_disc_old_filter.append(logp_disc_old_all[start:end])
                v_filter.append(v_all[start:end])
                logp_disc_pik_filter.append(logp_disc_pik_all[start:end])
                weights_filter.append(weights_all[start:end])
                rho_filter.append(rho_all[start:end])

        obs_filter = np.concatenate(obs_filter, axis=0)
        ac_filter = np.concatenate(ac_filter, axis=0)
        adv_filter = np.concatenate(adv_filter, axis=0)
        ret_filter = np.concatenate(ret_filter, axis=0)
        logp_disc_old_filter = np.concatenate(logp_disc_old_filter, axis=0)
        v_filter = np.concatenate(v_filter, axis=0)
        logp_disc_pik_filter = np.concatenate(logp_disc_pik_filter, axis=0)
        weights_filter = np.concatenate(weights_filter, axis=0)
        rho_filter = np.concatenate(rho_filter, axis=0)

        n_trajs_active = obs_filter.shape[0] // self.horizon

        on_policy = np.zeros_like(adv_filter)
        on_policy[:self.horizon] = n_trajs_active
        # self.count += 1
        # with open(f'/data/hanjl/debug_data/buf_data_{self.count}.pkl', 'wb') as f:
        #     pickle.dump(buf_data, f)

        # Filter tracj if bias

        lr = self.lr if self.fixed_lr else np.maximum(self.lr * frac, 1e-4)

        pi_loss_ctl_buf = []
        pi_loss_buf = []
        vf_loss_buf = []
        ent_buf = []
        kl_buf = []
        ratio_buf = []
        ratioclipfrac_buf = []
        gradclipped_buf = []

        length = obs_filter.shape[0]
        minibatch = length // self.nminibatches
        indices = np.arange(length)
        for _ in range(self.train_iters):
            # Randomize the indexes
            # np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            # for start in range(0, length, minibatch):
            for _ in range(self.nminibatches):
                minibatch_on = self.horizon // self.nminibatches
                idx_on = np.random.choice(indices[:self.horizon], minibatch_on)
                if length > self.horizon:
                    minibatch_off = (length - self.horizon) // self.nminibatches
                    idx_off = np.random.choice(indices[self.horizon:], minibatch_off)
                else:
                    idx_off = []
                mbinds = np.concatenate([idx_on, idx_off]).astype(np.int64)
                # end = start + minibatch
                # mbinds = indices[start:end]
                # slices = [arr[mbinds] for arr in buf_data]
                # [obs, actions, advs, rets, logprobs, values, rhos] = slices
                advs = adv_filter[mbinds]
                rhos = rho_filter[mbinds]
                rhos = np.minimum(rhos, 1.0)
                weights = weights_filter[mbinds]
                advs_mean = np.mean(advs * rhos * weights) / np.mean(rhos * weights)
                advs_std = np.std(advs * rhos * weights)
                advs_norm = (advs - advs_mean) / (advs_std + 1e-8)
                inputs = {
                    self.obs_ph: obs_filter[mbinds],
                    self.act_ph: ac_filter[mbinds],
                    self.adv_ph: advs_norm,
                    self.ret_ph: ret_filter[mbinds],
                    self.val_ph: v_filter[mbinds],
                    self.logp_disc_old_ph: logp_disc_old_filter[mbinds],
                    self.logp_disc_pik_ph: logp_disc_pik_filter[mbinds],
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
                ratio, ratioclipfrac, gradclipped = infos
                ratio_buf.append(ratio)
                ratioclipfrac_buf.append(ratioclipfrac)
                gradclipped_buf.append(gradclipped)

            tv_inputs = {
                self.obs_ph: obs_filter,
                self.act_ph: ac_filter,
                self.logp_disc_old_ph: logp_disc_old_filter,
                self.logp_disc_pik_ph: logp_disc_pik_filter,
                self.on_policy_ph: on_policy
            }
            pi_loss_ctl_all = self.sess.run(self.pi_loss_ctl, feed_dict=tv_inputs)

        # if tv_all > self.thresh * 0.5 * self.clip_ratio * 2:
        #     self.lr /= (1 + self.alpha)
        # elif tv_all < self.thresh * 0.5 * self.clip_ratio:
        #     self.lr *= (1 + self.alpha)
        pi_loss_ctl_mean = np.mean(pi_loss_ctl_buf)

        if pi_loss_ctl_mean > self.target_j * 1.5:
            self.alpha *= 2
        elif pi_loss_ctl_mean < self.target_j / 1.5:
            self.alpha /= 2
        self.alpha = np.clip(self.alpha, 2**(-10), 64)

        # Here you can add any information you want to log!
        if log2board:
            train_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag="loss/gradclipfrac", simple_value=np.mean(gradclipped)),
                tf.compat.v1.Summary.Value(
                    tag="loss/lr", simple_value=lr),
                tf.compat.v1.Summary.Value(
                    tag="loss/ratio", simple_value=np.mean(ratio_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/ratioclipfrac", simple_value=np.mean(ratioclipfrac_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/pi_loss_ctl", simple_value=pi_loss_ctl_mean),
                tf.compat.v1.Summary.Value(
                    tag="loss/trajs_active", simple_value=n_trajs_active),
                tf.compat.v1.Summary.Value(
                    tag="loss/absrho", simple_value=np.mean(meanabsrho_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/alpha", simple_value=self.alpha)
            ])
            self.summary_writer.add_summary(train_summary, step)

        return [np.mean(pi_loss_buf), np.mean(vf_loss_buf),
                np.mean(ent_buf), np.mean(kl_buf)]

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

    def compute_v_pik(self, obs):
        return self.sess.run(self.v, feed_dict={self.obs_ph: obs})

    def compute_logp_pik(self, obs, ac):
        inputs = {
            self.obs_ph: obs,
            self.act_ph: ac
        }
        return self.sess.run(self.logp_a_disc, feed_dict=inputs)
