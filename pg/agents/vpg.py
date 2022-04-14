import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

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


class VPGAgent():
    """
    The difference between Vanilla Policy Gradient (VPG) with a baseline as value
    function and Advantage Actor-Critic (A2C) is very similar to the difference
    between Monte Carlo Control and SARSA.
    """

    def __init__(self, sess, obs_dim, act_dim, num_env=1,
                 lr=3e-4, train_iters=10, ent_coef=0.0,
                 vf_coef=0.5, max_grad_norm=0.5, horizon=2048,
                 minibatch=64, gamma=0.99, lam=0.95,
                 fixed_lr=True):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_env = num_env
        self.lr = lr
        self.train_iters = train_iters
        self.horizon = horizon
        self.minibatch = minibatch * num_env
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.fixed_lr = fixed_lr

        self.buffer = Buffer.GAEBuffer(
            obs_dim, act_dim, size=horizon, num_env=num_env, gamma=gamma, lam=lam)
        self._build_network()
        self._build_train_op()
        self.saver = self._build_saver()

    # Note: Required to be called after _build_train_op(), otherwise return []
    def _get_var_list(self, name='pi'):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, name))
        return vars

    def _build_network(self):
        self.actor = ActorMLP(self.act_dim, name='pi')
        self.critic = CriticMLP(name='vf')

    def _build_train_op(self):
        ob1_ph = tf.compat.v1.placeholder(
            shape=(self.num_env, ) + self.obs_dim, dtype=tf.float32, name="ob1_ph")
        obs_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        act_ph = tf.compat.v1.placeholder(
            shape=(self.minibatch, ) + self.act_dim, dtype=tf.float32, name="act_ph")
        adv_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="adv_ph")
        ret_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="ret_ph")
        logp_old_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="logp_old_ph")
        val_ph = tf.compat.v1.placeholder(
            shape=[self.minibatch, ], dtype=tf.float32, name="val_ph")
        lr_ph = tf.compat.v1.placeholder(
            shape=[], dtype=tf.float32, name="lr_ph")

        all_phs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph, val_ph, lr_ph]

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
        entropy = tf.reduce_mean(dist.entropy())

        v = self.critic(obs_ph)

        # VPG objectives
        pi_loss = -tf.reduce_mean(adv_ph * logp_a)
        vf_loss = 0.5 * tf.reduce_mean(tf.square(v - ret_ph))

        # Usefull Infos
        approx_kl = 0.5 * tf.reduce_mean(tf.square(logp_old_ph - logp_a))

        # Total loss
        loss = pi_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        # Optimizers
        pi_optimizer = tf.train.AdamOptimizer(
            learning_rate=lr_ph, epsilon=1e-5)
        vf_optimizer = tf.train.AdamOptimizer(
            learning_rate=lr_ph, epsilon=1e-5)
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        pi_train_op = pi_optimizer.minimize(pi_loss, var_list=pi_params)
        vf_train_op = vf_optimizer.minimize(vf_loss, var_list=vf_params)

        self.ob1_ph = ob1_ph
        self.all_phs = all_phs
        self.get_action_ops = get_action_ops
        self.v1 = v1
        self.pi_loss = pi_loss
        self.vf_loss = vf_loss
        self.entropy = entropy
        self.approx_kl = approx_kl
        self.pi_train_op = pi_train_op
        self.vf_train_op = vf_train_op

    def update(self, frac):
        buf_data = self.buffer.get()
        assert buf_data[0].shape[0] == self.horizon * self.num_env

        pi_loss_buf = []
        vf_loss_buf = []
        entropy_buf = []
        kl_buf = []

        indices = np.arange(self.horizon * self.num_env)

        for i in range(self.train_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                slices = [arr[mbinds] for arr in buf_data]
                [obs, actions, advs, rets, logprobs, values] = slices
                # advs_raw = rets - values
                advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
                lr = self.lr if self.fixed_lr else self.lr * frac
                inputs = {
                    self.all_phs[0]: obs,
                    self.all_phs[1]: actions,
                    self.all_phs[2]: advs,
                    self.all_phs[3]: rets,
                    self.all_phs[4]: logprobs,
                    self.all_phs[5]: values,
                    self.all_phs[6]: lr,
                }

                if i == 1:
                    pi_loss, entropy, kl, _ = self.sess.run(
                        [self.pi_loss, self.entropy, self.approx_kl,
                         self.pi_train_op],
                        feed_dict=inputs)
                    pi_loss_buf.append(pi_loss)
                    entropy_buf.append(entropy)
                    kl_buf.append(kl)

                vf_loss, _ = self.sess.run(
                    [self.vf_loss, self.vf_train_op],
                    feed_dict=inputs)
                vf_loss_buf.append(vf_loss)

        return [np.mean(pi_loss_buf), np.mean(vf_loss_buf),
                np.mean(entropy_buf), np.mean(kl_buf)]

    def select_action(self, obs, deterministic=False):
        [mu, pi, v, logp_pi] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(self.num_env, -1)})
        self.extra_info = [v, logp_pi]
        ac = mu if deterministic else pi
        return pi

    def compute_v(self, obs):
        return self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs.reshape(self.num_env, -1)})

    def store_transition(self, obs, action, reward, done):
        [v, logp_pi] = self.extra_info
        self.buffer.store(obs, action, reward, done,
                          v, logp_pi)

    def _build_saver(self):
        params = self._get_var_list('pi') + self._get_var_list('vf')
        return tf.compat.v1.train.Saver(var_list=params,
                                        max_to_keep=4)

    def bundle(self, checkpoint_dir, epoch):
        if not os.path.exists(checkpoint_dir):
            raise
        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=epoch)

    def unbundle(self, checkpoint_dir, epoch=None):
        if not os.path.exists(checkpoint_dir):
            raise
        self.saver.restore(
            self.sess,
            os.path.join(checkpoint_dir, f'tf_ckpt-{epoch}'))
