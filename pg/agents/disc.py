import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from pg.agents.base import BaseAgent
import pg.buffer.gaebuffer as Buffer


class DISCBuffer:
    '''
    Openai spinningup implementation
    '''

    def __init__(self, obs_dim, act_dim, size,  gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, ) + obs_dim, dtype=np.float64)
        self.act_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros((size, ), dtype=np.float32)
        self.rew_buf = np.zeros((size, ), dtype=np.float32)
        self.done_buf = np.zeros((size, ), dtype=np.float32)
        self.ret_buf = np.zeros((size, ), dtype=np.float32)
        self.val_buf = np.zeros((size+1, ), dtype=np.float32)
        self.logp_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.path_start_idx = 0

    def store(self, obs, act, rew, done, val, logp):
        assert self.ptr < self.max_size
        assert obs.shape == self.obs_dim
        assert act.shape == self.act_dim
        assert logp.shape == self.act_dim
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=None):
        self.val_buf[self.ptr] = last_val
        path_slice = slice(self.path_start_idx, self.ptr)

        # GAE-Lambda advantage calculation
        lastgaelam = 0.0
        for t in reversed(range(self.path_start_idx, self.ptr)):
            delta = self.rew_buf[t] + \
                self.gamma * self.val_buf[t + 1] - self.val_buf[t]
            self.adv_buf[t] = lastgaelam = \
                delta + self.gamma * self.lam * lastgaelam

        self.ret_buf[path_slice] = self.adv_buf[path_slice] + \
            self.val_buf[path_slice]

        self.path_start_idx = self.ptr
        pass

    def get(self):
        assert self.ptr == self.max_size
        return [self.obs_buf,
                self.act_buf,
                self.adv_buf,
                self.ret_buf,
                self.logp_buf,
                self.val_buf[:-1]]

    def get_rms_data(self):
        assert self.ptr == self.max_size
        return [self.obs_buf,
                self.ret_buf]

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0


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
                 clip_ratio=0.2, lr=3e-4, train_iters=10, target_kl=0.01,
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_j=1e-3,
                 horizon=2048, minibatch=64, gamma=0.99, lam=0.95,
                 alpha=1, grad_clip=True, vf_clip=True, fixed_lr=False):
        self.sess = sess
        self.summary_writer = summary_writer
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
        self.target_j = target_j
        self.alpha = alpha
        self.grad_clip = grad_clip
        self.vf_clip = vf_clip
        self.fixed_lr = fixed_lr

        self.buffer = DISCBuffer(
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
            shape=(self.minibatch, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        self.act_ph = act_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.act_dim, dtype=tf.float32, name="act_ph")
        self.adv_ph = adv_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="adv_ph")
        self.ret_ph = ret_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="ret_ph")
        self.logp_old_ph = logp_old_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.act_dim, dtype=tf.float32, name="logp_old_ph")
        self.val_ph = val_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="val_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")
        self.alpha_ph = alpha_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="alpha_ph")

        # Probability distribution
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.act_dim[0]),
            initializer=tf.zeros_initializer)
        std = tf.exp(logstd)

        # Interative with env
        mu1 = self.actor(ob1_ph)
        dist1 = tfp.distributions.Normal(loc=mu1, scale=std)
        pi1 = dist1.sample()
        # logp_pi1 = tf.reduce_sum(dist1.log_prob(pi1), axis=1)
        logp_pi1 = dist1.log_prob(pi1)

        v1 = self.critic(ob1_ph)

        get_action_ops = [mu1, pi1, v1, logp_pi1]

        # Train batch data
        mu = self.actor(obs_ph)
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        # logp_a = tf.reduce_sum(dist.log_prob(act_ph), axis=1)
        logp_a = dist.log_prob(act_ph)
        entropy = tf.reduce_sum(dist.entropy(), axis=1)
        meanent = tf.reduce_mean(entropy)

        v = self.critic(obs_ph)

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio_disc = tf.exp(logp_a - logp_old_ph)
        ratio_disc2 = tf.clip_by_value(
            ratio_disc, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        # ratio = tf.reduce_prod(ratio_disc, axis=1)
        # ratio2 = tf.reduce_prod(ratio_disc2, axis=1)
        min_ratio = tf.where(
            adv_ph > 0,
            tf.minimum(ratio_disc, ratio_disc2),
            tf.maximum(ratio_disc, ratio_disc2))
        ratio = tf.reduce_prod(ratio_disc, axis=1)
        ratio2 = tf.reduce_prod(min_ratio, axis=1)

        # pi_loss1 = -adv_ph * ratio
        # pi_loss2 = -adv_ph * tf.clip_by_value(
        #     ratio2, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        # pi_loss = tf.reduce_mean(tf.maximum(pi_loss1, pi_loss2))
        pi_loss_ctl = 0.5 * tf.reduce_mean(tf.square(tf.log(ratio)))

        pi_loss = -tf.reduce_mean(adv_ph * ratio2) + alpha_ph * pi_loss_ctl

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
        logp_old = tf.reduce_sum(logp_old_ph, axis=1)
        lopp_new = tf.reduce_sum(logp_a, axis=1)
        meankl = 0.5 * tf.reduce_mean(tf.square(logp_old - lopp_new))
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
            gradclipped = _grad_norm > self.max_grad_norm
        grads_and_vars = list(zip(grads, vars))

        train_op = optimizer.apply_gradients(grads_and_vars)

        self.get_action_ops = get_action_ops
        self.v1 = v1
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

    def update(self, frac, log2board, step):
        buf_data = self.buffer.get()
        assert buf_data[0].shape[0] == self.horizon

        lr = self.lr if self.fixed_lr else self.lr * frac

        pi_loss_buf = []
        vf_loss_buf = []
        ent_buf = []
        kl_buf = []
        ratio_buf = []
        ratioclipfrac_buf = []
        gradclipped_buf = []
        pi_loss_ctl_buf = []
        alpha_buf = []

        indices = np.arange(self.horizon)
        for _ in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                slices = [arr[mbinds] for arr in buf_data]
                [obs, actions, advs, rets, logprobs, values] = slices
                advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
                inputs = {
                    self.obs_ph: obs,
                    self.act_ph: actions,
                    self.adv_ph: advs,
                    self.ret_ph: rets,
                    self.logp_old_ph: logprobs,
                    self.val_ph: values,
                    self.lr_ph: lr,
                    self.alpha_ph: self.alpha
                }

                infos, losses, _ = self.sess.run(
                    [self.infos, self.losses, self.train_op], feed_dict=inputs)
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

        meanctl = np.mean(pi_loss_ctl_buf)
        if meanctl < self.target_j / 1.5:
            self.alpha /= 2
        elif meanctl > self.target_j * 1.5:
            self.alpha *= 2
        self.alpha = np.clip(self.alpha, 2**(-10), 64)
        alpha_buf.append(self.alpha)

        # Here you can add any information you want to log!
        if log2board:
            train_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag="loss/gradclipfrac", simple_value=np.mean(gradclipped_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/lr", simple_value=lr),
                tf.compat.v1.Summary.Value(
                    tag="loss/ratio", simple_value=np.mean(ratio_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/ratioclipfrac", simple_value=np.mean(ratioclipfrac_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/alpha", simple_value=np.mean(alpha_buf))
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
