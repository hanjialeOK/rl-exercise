import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import pickle

from pg.agents.base import BaseAgent
import pg.buffer.gaebuffer as Buffer


def tf_ortho_init(scale):
    return tf.keras.initializers.Orthogonal(scale)

def flat(var_list):
    return tf.concat([tf.reshape(v, (-1,)) for v in var_list], axis=0)

def setfromflat(var_list, theta):
    shapes = [v.get_shape().as_list() for v in var_list]
    assigns = []
    start = 0
    for (v, shape) in zip(var_list, shapes):
        size = int(np.prod(shape))
        new = theta[start:start+size]
        assigns.append(tf.assign(v, tf.reshape(new, shape)))
        start += size
    return assigns

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
                 ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=2048, nminibatches=32, gamma=0.99, lam=0.95,
                 grad_clip=True, vf_clip=False, fixed_lr=False,
                 thresh=0.5, alpha=0.03, nlatest=8):
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
        self.grad_clip = grad_clip
        self.vf_clip = vf_clip
        self.fixed_lr = fixed_lr
        self.thresh = thresh
        self.alpha = alpha
        self.nlatest = nlatest

        self.buffer = Buffer.GAEVBuffer(
            obs_dim, act_dim, size=horizon, nlatest=nlatest, gamma=gamma, lam=lam)
        self._build_network()
        self._build_train_op()
        self.saver = self._build_saver()

    def _build_network(self):
        self.actor = ActorMLP(self.act_dim, name='pi')
        # self.actor_pik = ActorMLP(self.act_dim, name='pik')
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
        self.logp_pik_ph = logp_pik_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="logp_pik_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")

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

        # PPO objectives
        # pi(a|s) / pi_old(a|s), should be one at the first iteration
        ratio = tf.exp(logp_a - logp_old_ph)
        ratio_pik = tf.exp(logp_pik_ph - logp_old_ph)
        pi_loss1 = -adv_ph * ratio
        pi_loss2 = -adv_ph * tf.clip_by_value(
            ratio, ratio_pik - self.clip_ratio, ratio_pik + self.clip_ratio)
        pi_loss = tf.reduce_mean(tf.maximum(pi_loss1, pi_loss2))

        tv = 0.5 * tf.reduce_mean(tf.abs(ratio - ratio_pik))

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
        meankl = 0.5 * tf.reduce_mean(tf.square(logp_pik_ph - logp_a))
        absratio = tf.reduce_mean(tf.abs(ratio - ratio_pik) + 1.0)
        ratioclipped = tf.where(
            adv_ph > 0,
            ratio > (ratio_pik + self.clip_ratio),
            ratio < (ratio_pik - self.clip_ratio))
        ratioclipfrac = tf.reduce_mean(tf.cast(ratioclipped, tf.float32))

        # Total loss
        loss = pi_loss - meanent * self.ent_coef + vf_loss * self.vf_coef

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
            gradclipped = _grad_norm > self.max_grad_norm
        grads_and_vars = list(zip(grads, vars))

        pi_train_op = pi_optimizer.apply_gradients(grads_and_vars)
        vf_train_op = vf_optimizer.minimize(vf_loss, var_list=vf_params)

        self.get_action_ops = get_action_ops
        self.logp_a = logp_a
        self.v1 = v1
        self.v = v
        self.absratio = absratio
        self.pi_loss = pi_loss
        self.vf_loss = vf_loss
        self.meankl = meankl
        self.meanent = meanent
        self.tv = tv
        self.ratioclipfrac = ratioclipfrac
        self.gradclipped = gradclipped
        # self.train_op = train_op
        self.pi_train_op = pi_train_op
        self.vf_train_op = vf_train_op

        self.losses = [pi_loss, vf_loss, meanent, meankl, tv]
        self.infos = [absratio, ratioclipfrac, gradclipped]
        self.count = 0

        # self.pi_flatted = flat(pi_params)
        # self.vf_flatted = flat(vf_params)

    def update(self, frac, log2board, step):
        # obs_all, obs2_all, ac_all = self.buffer.get_obs()
        # v_pik = self.sess.run(self.v, feed_dict={ self.obs_ph: obs_all })
        # v2_pik = self.sess.run(self.v, feed_dict={ self.obs_ph: obs2_all })
        # logp_inputs = {self.obs_ph: obs_all, self.act_ph: ac_all}
        # logp_pik = self.sess.run(self.logp_a, feed_dict=logp_inputs)
        buf_data = self.buffer.vtrace(self.compute_v_pik, self.compute_logp_pik)
        self.buffer.update()
        [obs_all, ac_all, adv_all, ret_all, logp_old_all, v_all, logp_pik_all] = buf_data
        rho_all = np.exp(logp_pik_all - logp_old_all)

        # self.count += 1
        # with open(f'/data/hanjl/debug_data/buf_data_{self.count}.pkl', 'wb') as f:
        #     pickle.dump(buf_data, f)

        lr = self.lr if self.fixed_lr else self.lr * frac

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
                # slices = [arr[mbinds] for arr in buf_data]
                # [obs, actions, advs, rets, logprobs, values, rhos] = slices
                advs = adv_all[mbinds]
                rhos = rho_all[mbinds]
                advs_mean = np.mean(advs * rhos) / np.mean(rhos)
                advs_std = np.std(advs * rhos)
                advs_norm = (advs - advs_mean) / (advs_std + 1e-8)
                inputs = {
                    self.obs_ph: obs_all[mbinds],
                    self.act_ph: ac_all[mbinds],
                    self.adv_ph: advs_norm,
                    self.ret_ph: ret_all[mbinds],
                    self.val_ph: v_all[mbinds],
                    self.logp_old_ph: logp_old_all[mbinds],
                    self.logp_pik_ph: logp_pik_all[mbinds],
                    self.lr_ph: self.lr
                }

                infos, losses, _, _ = self.sess.run(
                    [self.infos, self.losses, self.pi_train_op, self.vf_train_op],
                    feed_dict=inputs)
                # Unpack losses
                pi_loss, vf_loss, ent, kl, tv = losses
                pi_loss_buf.append(pi_loss)
                vf_loss_buf.append(vf_loss)
                ent_buf.append(ent)
                kl_buf.append(kl)
                tv_buf.append(tv)
                # Unpack infos
                ratio, ratioclipfrac, gradclipped = infos
                ratio_buf.append(ratio)
                ratioclipfrac_buf.append(ratioclipfrac)
                gradclipped_buf.append(gradclipped)

            tv_inputs = {
                self.obs_ph: obs_all,
                self.act_ph: ac_all,
                self.logp_old_ph: logp_old_all,
                self.logp_pik_ph: logp_pik_all
            }
            tv_all = self.sess.run(self.tv, feed_dict=tv_inputs)

        if tv_all > self.thresh * 0.5 * self.clip_ratio * 2:
            self.lr /= (1 + self.alpha)
        elif tv_all < self.thresh * 0.5 * self.clip_ratio:
            self.lr *= (1 + self.alpha)

        # Here you can add any information you want to log!
        if log2board:
            train_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag="loss/gradclipfrac", simple_value=np.mean(gradclipped)),
                tf.compat.v1.Summary.Value(
                    tag="loss/lr", simple_value=self.lr),
                tf.compat.v1.Summary.Value(
                    tag="loss/ratio", simple_value=np.mean(ratio_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/ratioclipfrac", simple_value=np.mean(ratioclipfrac_buf)),
                tf.compat.v1.Summary.Value(
                    tag="loss/tv", simple_value=tv_all)
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
        return self.sess.run(self.logp_a, feed_dict=inputs)

    # def setactorfromflat(self):
    #     var_list = self._get_var_list('pi')
    #     x = flat(var_list)
    #     self.actor_param_ph = tf.compat.v1.placeholder(
    #         shape=x.shape, dtype=tf.float32, name="actor_param_ph")
    #     assigns = setfromflat(var_list, self.actor_param_ph)
    #     return assigns

    # def setcriticfromflat(self):
    #     var_list = self._get_var_list('vf')
    #     x = flat(var_list)
    #     self.critic_param_ph = tf.compat.v1.placeholder(
    #         shape=x.shape, dtype=tf.float32, name="critic_param_ph")
    #     assigns = setfromflat(var_list, self.critic_param_ph)
    #     return assigns

    # def assign_actor_weights(self, param):
    #     self.sess.run(self.actor_assign, feed_dict={self.actor_param_ph: param})

    # def assign_critic_weights(self, param):
    #     self.sess.run(self.critic_assign, feed_dict={self.critic_param_ph: param})
