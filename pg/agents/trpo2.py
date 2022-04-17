import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os

from pg.agents.base import BaseAgent
import pg.buffer.gaebuffer as Buffer

from termcolor import cprint

EPS = 1e-8


def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose:
        print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))
    return x


def flat(var_list):
    return tf.concat([tf.reshape(v, (-1,)) for v in var_list], axis=0)


def flatgrad(loss, var_list):
    return flat(tf.compat.v1.gradients(ys=loss, xs=var_list))


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


class TRPOAgent(BaseAgent):
    def __init__(self, sess, obs_dim, act_dim, num_env=1,
                 max_kl=0.01, vf_lr=1e-3, train_vf_iters=5, ent_coef=0.0,
                 cg_damping=0.1, cg_iters=10, horizon=1024, minibatch=64,
                 gamma=0.99, lam=0.98):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_env = num_env
        self.max_kl = max_kl
        self.vf_lr = vf_lr
        self.train_vf_iters = train_vf_iters
        self.ent_coef = ent_coef
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.horizon = horizon
        self.minibatch = minibatch * num_env

        self.buffer = Buffer.GAEBuffer(
            obs_dim, act_dim, size=horizon, num_env=num_env, gamma=gamma, lam=lam)
        self._build_network()
        self._build_train_op()
        self.saver = self._build_saver()
        self.sync_ops = self._build_sync_op()

    def _build_network(self):
        self.actor = ActorMLP(self.act_dim, name='pi')
        self.critic = CriticMLP(name='vf')
        self.actor_old = ActorMLP(self.act_dim, name='pi_old')

    def _build_train_op(self):
        self.ob1_ph = ob1_ph = tf.compat.v1.placeholder(
            shape=(self.num_env, ) + self.obs_dim, dtype=tf.float32, name="ob1_ph")
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

        # Probability distribution
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd', shape=(1, self.act_dim[0]),
            initializer=tf.zeros_initializer)
        logstd_old = tf.compat.v1.get_variable(
            name='pi_old/logstd', shape=(1, self.act_dim[0]),
            initializer=tf.zeros_initializer)
        std = tf.exp(logstd)
        std_old = tf.stop_gradient(tf.exp(logstd_old))

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

        mu_old = tf.stop_gradient(self.actor_old(obs_ph))
        dist_old = tfp.distributions.Normal(loc=mu_old, scale=std_old)
        kl = tf.reduce_sum(dist.kl_divergence(dist_old), axis=1)
        meankl = tf.reduce_mean(kl)

        v = self.critic(obs_ph)

        # TRPO objectives
        ratio = tf.exp(logp_a - logp_old_ph)
        surrgain = tf.reduce_mean(ratio * adv_ph)
        vf_loss = 0.5 * tf.reduce_mean(tf.square(v - ret_ph))
        optimgain = surrgain + self.ent_coef * meanent

        # Value function optimizer
        vf_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.vf_lr, epsilon=1e-8)
        vf_params = self._get_var_list('vf')
        vf_train_op = vf_optimizer.minimize(vf_loss, var_list=vf_params)

        # Symbols needed for CG solver
        pi_params = self._get_var_list('pi')
        surr_grads_flatted = flatgrad(optimgain, pi_params)
        kl_grads_flatted = flatgrad(meankl, pi_params)
        tangent_ph = tf.compat.v1.placeholder(
            shape=kl_grads_flatted.shape,
            dtype=tf.float32, name='tangent_ph')
        fvpbase = flatgrad(tf.reduce_sum(
            kl_grads_flatted * tangent_ph), pi_params)
        fvp = fvpbase + self.cg_damping * tangent_ph

        # Symbols for getting and setting params
        pi_params_flatted = flat(pi_params)
        newv_ph = tf.compat.v1.placeholder(
            shape=pi_params_flatted.shape,
            dtype=tf.float32, name="newv_ph")
        set_pi_params = setfromflat(pi_params, newv_ph)

        self.tangent_ph = tangent_ph
        self.newv_ph = newv_ph
        self.get_action_ops = get_action_ops
        self.v1 = v1
        self.surrgain = surrgain
        self.vf_loss = vf_loss
        self.optimgain = optimgain
        self.kl = meankl
        self.entropy = meanent
        self.vf_train_op = vf_train_op
        self.surr_grads_flatted = surr_grads_flatted
        self.pi_params_flatted = pi_params_flatted
        self.fvp = fvp
        self.set_pi_params = set_pi_params

    def _build_sync_op(self):
        sync_qt_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_old = self._get_var_list('pi_old')

        for (newv, oldv) in zip(pi_params, pi_params_old):
            sync_qt_ops.append(oldv.assign(newv, use_locking=True))
        return sync_qt_ops

    def update(self, frac=None):
        buf_data = self.buffer.get()
        assert buf_data[0].shape[0] == self.horizon * self.num_env
        [obs, actions, advs, rets, logprobs, values] = buf_data
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

        surrgain_buf = []
        vf_loss_buf = []
        entropy_buf = []
        kl_buf = []

        inputs = {
            self.obs_ph: obs,
            self.act_ph: actions,
            self.adv_ph: advs,
            self.logp_old_ph: logprobs
        }
        fvp_inputs = {
            self.obs_ph: obs[::5]
        }

        def fisher_vector_product(x):
            return self.sess.run(
                self.fvp, feed_dict={**fvp_inputs, self.tangent_ph: x})

        self.sess.run(self.sync_ops)
        g = self.sess.run(
            self.surr_grads_flatted, feed_dict=inputs)
        if np.allclose(g, 0):
            print("Got zero gradient. not updating")
        else:
            stepdir = cg(fisher_vector_product, g, self.cg_iters)
            assert np.isfinite(stepdir).all()
            shs = .5 * np.dot(stepdir, fisher_vector_product(stepdir))
            lm = np.sqrt(shs / self.max_kl)
            fullstep = stepdir / lm
            expectedimprove = np.dot(g, fullstep)
            oldv = self.sess.run(self.pi_params_flatted)
            surrbefore = self.sess.run(self.optimgain, feed_dict=inputs)
            stepsize = 1.0
            for i in range(10):
                newv = oldv + fullstep * stepsize
                self.sess.run(self.set_pi_params,
                              feed_dict={self.newv_ph: newv})
                surr, kl, entropy = self.sess.run(
                    [self.optimgain, self.kl, self.entropy], feed_dict=inputs)
                surrgain_buf.append(surr)
                kl_buf.append(kl)
                entropy_buf.append(entropy)
                improve = surr - surrbefore
                print("Expected: %.3f Actual: %.3f" %
                      (expectedimprove, improve))
                if not np.isfinite([surr, kl]).all():
                    cprint("Got non-finite value of losses -- bad!",
                           color='red')
                elif kl > self.max_kl * 1.5:
                    cprint("violated KL constraint. shrinking step.",
                           color='yellow')
                elif improve < 0:
                    cprint("surrogate didn't improve. shrinking step.",
                           color='yellow')
                else:
                    cprint("Stepsize OK!", color='green')
                    break
                stepsize *= 0.5
            else:
                cprint("couldn't compute a good step",
                       color='red', attrs=['bold'])
                self.sess.run(self.set_pi_params,
                              feed_dict={self.newv_ph: oldv})

        indices = np.arange(self.horizon * self.num_env)
        for _ in range(self.train_vf_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                batch_inputs = {
                    self.obs_ph: obs[mbinds],
                    self.ret_ph: rets[mbinds]
                }
                vf_loss, _ = self.sess.run(
                    [self.vf_loss, self.vf_train_op],
                    feed_dict=batch_inputs)
                vf_loss_buf.append(vf_loss)

        return [np.mean(surrgain_buf), np.mean(vf_loss_buf),
                np.mean(entropy_buf), np.mean(kl_buf), self.vf_lr]

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
