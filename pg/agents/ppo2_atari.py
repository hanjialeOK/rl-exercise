import numpy as np
import tensorflow as tf

import pg.buffer.gaebuffer_atari as Buffer
from pg.agents.base import BaseAgent
from common.distributions import CategoricalPd


def tf_ortho_init(scale):
    return tf.keras.initializers.Orthogonal(scale)


class ActorCriticCNN(tf.keras.Model):
    def __init__(self, ac_dim, name=None):
        super(ActorCriticCNN, self).__init__(name=name)
        activation_fn = tf.keras.activations.relu
        kernel_initializer = None
        self.conv1 = tf.keras.layers.Conv2D(
            32, [8, 8], strides=4, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='conv1')
        self.conv2 = tf.keras.layers.Conv2D(
            64, [4, 4], strides=2, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='conv2')
        self.conv3 = tf.keras.layers.Conv2D(
            64, [3, 3], strides=1, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='conv3')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            512, activation=activation_fn,
            kernel_initializer=tf_ortho_init(np.sqrt(2)), name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            ac_dim, activation=None,
            kernel_initializer=tf_ortho_init(np.sqrt(0.01)), name='pi')
        self.dense3 = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=tf_ortho_init(1.0), name='vf')

    def call(self, state):
        x = tf.cast(state, tf.float32) / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        v = self.dense3(x)
        return logits, tf.squeeze(v, axis=-1)


class PPOAgent(BaseAgent):
    def __init__(self, sess, env, nenv=8,
                 clip_ratio=0.1, lr=2.5e-4, train_iters=4, target_kl=0.01,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 horizon=128, nminibatches=4, gamma=0.99, lam=0.95,
                 grad_clip=True, vf_clip=True, fixed_lr=False):
        self.sess = sess
        self.obs_shape = (84, 84, 4)
        self.ac_shape = env.action_space.n
        self.obs_dtype = env.observation_space.dtype.name
        self.ac_dtype = env.action_space.dtype.name
        self.nenv = nenv
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
        self.nbatch = horizon * nenv

        self.buffer = Buffer.GAEBuffer(
            env, nenv=nenv, horizon=horizon, gamma=gamma, lam=lam,
            compute_v=self.compute_v)
        self._build_network()
        self._build_train_op()
        self.sync_op = self._build_sync_op()
        super().__init__()

    def _build_network(self):
        self.actorcritic = ActorCriticCNN(self.ac_shape, name='ppo')
        self.actorcritic_pik = ActorCriticCNN(self.ac_shape, name='ppo_pik')

    def _build_train_op(self):
        self.ob1_ph = ob1_ph = tf.compat.v1.placeholder(
            shape=(self.nenv, ) + self.obs_shape, dtype=self.obs_dtype, name="ob1_ph")
        self.obs_ph = obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_shape, dtype=self.obs_dtype, name="obs_ph")
        self.ac_ph = ac_ph = tf.compat.v1.placeholder(
            shape=(None, ), dtype=self.ac_dtype, name="ac_ph")
        self.adv_ph = adv_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="adv_ph")
        self.ret_ph = ret_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="ret_ph")
        self.neglogp_old_ph = neglogp_old_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="neglogp_old_ph")
        self.val_ph = val_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="val_ph")
        self.lr_ph = lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")

        # Probability distribution

        # Interative with env
        mu1, v1 = self.actorcritic(ob1_ph)
        dist1 = CategoricalPd(mu1)
        pi1 = dist1.sample()
        neglogp1 = dist1.neglogp(pi1)

        get_action_ops = [mu1, pi1, v1, neglogp1]

        # Train batch data
        mu, v = self.actorcritic(obs_ph)
        dist = CategoricalPd(mu)
        neglogpac = dist.neglogp(ac_ph)
        entropy = dist.entropy()
        meanent = tf.reduce_mean(entropy)

        mu_pik, _ = self.actorcritic_pik(obs_ph)
        mu_pik = tf.stop_gradient(mu_pik)
        dist_pik = CategoricalPd(mu_pik)
        kl = dist_pik.kl(dist)
        meankl = tf.reduce_mean(kl)

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
        loss = pi_loss - meanent * self.ent_coef + vf_loss * self.vf_coef

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
        params = self._get_var_list('ppo')
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
        self.train_op = train_op

        self.stats_list = [pi_loss, vf_loss, meanent, approxkl, meankl, absratio, ratioclipfrac, gradclipped]
        self.loss_names = ['pi_loss', 'vf_loss', 'entropy', 'kl', 'kl2', 'absratio', 'ratioclipfrac', 'gradclipped']
        assert len(self.stats_list) == len(self.loss_names)

    def _build_sync_op(self):
        sync_qt_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_old = self._get_var_list('pik')

        for (newv, oldv) in zip(pi_params, pi_params_old):
            sync_qt_ops.append(oldv.assign(newv, use_locking=True))
        return sync_qt_ops

    def update(self, frac, logger):
        buf_data = self.buffer.get()
        [obs_all, ac_all, adv_all, ret_all, val_all, neglogp_all] = buf_data
        assert obs_all.shape[0] == self.nbatch

        # lr = self.lr if self.fixed_lr else self.lr * frac
        lr = self.lr if self.fixed_lr else np.maximum(self.lr * frac, 1e-4)

        self.sess.run(self.sync_op)

        indices = np.arange(self.nbatch)
        minibatch = self.nbatch // self.nminibatches

        mblossvals = []
        for _ in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.nbatch, minibatch):
                end = start + minibatch
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
                    self.lr_ph: lr,
                }

                losses = self.sess.run(self.stats_list + [self.train_op], feed_dict=inputs)[:-1]
                mblossvals.append(losses)

        # Here you can add any information you want to log!
        lossvals = np.mean(mblossvals, axis=0)
        for (lossval, lossname) in zip(lossvals, self.loss_names):
            logger.logkv('loss/' + lossname, lossval)
        logger.logkv("loss/lr", lr)
        logger.logkv("loss/train_iter", self.train_iters)

    def select_action(self, obs, deterministic=False):
        [mu, pi, v, neglogp] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs})
        ac = mu if deterministic else pi
        return pi, v, neglogp, mu

    def compute_v(self, obs):
        v = self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs})
        return v
