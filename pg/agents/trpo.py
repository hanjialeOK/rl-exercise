import numpy as np
import tensorflow as tf
from common.distributions import DiagGaussianPd

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


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.compat.v1.gradients(ys=loss, xs=var_list)
    is_gradclipped = False
    if clip_norm is not None:
        grads, _grad_norm = tf.clip_by_global_norm(grads, clip_norm)
        is_gradclipped = _grad_norm > clip_norm
    return flat(grads), is_gradclipped


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
    def __init__(self, sess, env, 
                 max_kl=0.01, vf_lr=1e-3, train_vf_iters=5, ent_coef=0.0,
                 cg_damping=0.1, cg_iters=10, horizon=1024, minibatch=64,
                 gamma=0.99, lam=0.98):
        self.sess = sess
        self.obs_dim = env.observation_space.shape
        self.ac_dim = env.action_space.shape
        self.max_kl = max_kl
        self.vf_lr = vf_lr
        self.train_vf_iters = train_vf_iters
        self.ent_coef = ent_coef
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.horizon = horizon
        self.minibatch = minibatch

        self.buffer = Buffer.GAEBuffer(
            env, horizon=horizon, gamma=gamma, lam=lam,
            compute_v=self.compute_v)
        self._build_network()
        self._build_train_op()
        self.sync_ops = self._build_sync_op()
        super().__init__()

    def _build_network(self):
        self.actor = ActorMLP(self.ac_dim, name='pi')
        self.critic = CriticMLP(name='vf')
        self.actor_old = ActorMLP(self.ac_dim, name='pi_old')

    def _build_train_op(self):
        self.ob1_ph = ob1_ph = tf.compat.v1.placeholder(
            shape=(1, ) + self.obs_dim, dtype=tf.float32, name="ob1_ph")
        self.obs_ph = obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        self.ac_ph = ac_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.ac_dim, dtype=tf.float32, name="ac_ph")
        self.adv_ph = adv_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="adv_ph")
        self.ret_ph = ret_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="ret_ph")
        self.neglogp_old_ph = neglogp_old_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="neglogp_old_ph")
        self.val_ph = val_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="val_ph")

        # Probability distribution
        logstd1 = tf.compat.v1.get_variable(
            name='pi/logstd1', shape=(1, self.ac_dim[0]),
            initializer=tf.zeros_initializer)
        logstd1_old = tf.compat.v1.get_variable(
            name='pi_old/logstd1', shape=(1, self.ac_dim[0]),
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

        mu_old = tf.stop_gradient(self.actor_old(obs_ph))
        logstd_old = tf.tile(logstd1_old, (tf.shape(mu_old)[0], 1))
        dist_old = DiagGaussianPd(mu_old, logstd_old)
        kl = tf.reduce_sum(dist_old.kl(dist), axis=1)
        meankl = tf.reduce_mean(kl)

        v = self.critic(obs_ph)

        # TRPO objectives
        ratio = tf.exp(neglogp_old_ph - neglogpac)
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
        surr_grads_flatted, _ = flatgrad(optimgain, pi_params)
        kl_grads_flatted, _ = flatgrad(meankl, pi_params)
        tangent_ph = tf.compat.v1.placeholder(
            shape=kl_grads_flatted.shape,
            dtype=tf.float32, name='tangent_ph')
        fvpbase, _ = flatgrad(tf.reduce_sum(
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
        self.optimgain = optimgain
        self.vf_loss = vf_loss
        self.vf_train_op = vf_train_op
        self.surr_grads_flatted = surr_grads_flatted
        # self.is_gradclipped = is_gradclipped
        self.pi_params_flatted = pi_params_flatted
        self.fvp = fvp
        self.set_pi_params = set_pi_params

        self.pi_losses = [optimgain, meankl, surrgain, meanent]
        self.pi_loss_names = ['optimgain', 'kl', 'surrgain', 'entropy']
        self.vf_losses = [vf_loss]
        self.vf_loss_names = ['vf_loss']

    def _build_sync_op(self):
        sync_qt_ops = []
        pi_params = self._get_var_list('pi')
        pi_params_old = self._get_var_list('pi_old')

        for (newv, oldv) in zip(pi_params, pi_params_old):
            sync_qt_ops.append(oldv.assign(newv, use_locking=True))
        return sync_qt_ops

    def update(self, frac, logger):
        buf_data = self.buffer.get()
        assert buf_data[0].shape[0] == self.horizon
        [obs_all, ac_all, adv_all, ret_all, val_all, neglogp_all] = buf_data
        advs = (adv_all - np.mean(adv_all)) / (np.std(adv_all) + 1e-8)

        inputs = {
            self.obs_ph: obs_all,
            self.ac_ph: ac_all,
            self.adv_ph: advs,
            self.neglogp_old_ph: neglogp_all
        }
        fvp_inputs = {
            self.obs_ph: obs_all[::5]
        }

        def fisher_vector_product(x):
            return self.sess.run(
                self.fvp, feed_dict={**fvp_inputs, self.tangent_ph: x})

        self.sess.run(self.sync_ops)
        g = self.sess.run(
            self.surr_grads_flatted, feed_dict=inputs)
        if np.allclose(g, 0):
            print("Got zero gradient. not updating")
            for lossname in self.pi_loss_names:
                logger.logkv('loss/' + lossname, np.nan)
            logger.logkv('loss/linesearch', 0)
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
            mblossvals = []
            for i in range(10):
                newv = oldv + fullstep * stepsize
                self.sess.run(self.set_pi_params,
                              feed_dict={self.newv_ph: newv})
                lossvals = self.sess.run(self.pi_losses, feed_dict=inputs)
                mblossvals.append(lossvals)
                surr, kl = lossvals[0], lossvals[1]
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

            lossvals = np.mean(mblossvals, axis=0)
            for (lossval, lossname) in zip(lossvals, self.pi_loss_names):
                logger.logkv('loss/' + lossname, lossval)
            logger.logkv('loss/linesearch', len(mblossvals))

        indices = np.arange(self.horizon)
        mblossvals = []
        for _ in range(self.train_vf_iters):
            # Randomize the indexes
            np.random.shuffle(indices)
            # 0 to batch_size with batch_train_size step
            for start in range(0, self.horizon, self.minibatch):
                end = start + self.minibatch
                mbinds = indices[start:end]
                batch_inputs = {
                    self.obs_ph: obs_all[mbinds],
                    self.ret_ph: ret_all[mbinds]
                }
                vf_loss = self.sess.run(self.vf_losses + [self.vf_train_op], feed_dict=batch_inputs)[:-1]
                mblossvals.append(vf_loss)

        lossvals = np.mean(mblossvals, axis=0)
        for (lossval, lossname) in zip(lossvals, self.vf_loss_names):
            logger.logkv('loss/' + lossname, lossval)
        logger.logkv("loss/lr", self.vf_lr)

    def select_action(self, obs, deterministic=False):
        [mu, logstd, pi, v, neglogp] = self.sess.run(
            self.get_action_ops, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        ac = mu if deterministic else pi
        return ac, v, neglogp, mu, logstd

    def compute_v(self, obs):
        v = self.sess.run(
            self.v1, feed_dict={self.ob1_ph: obs.reshape(1, -1)})
        return v
