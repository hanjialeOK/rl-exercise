import numpy as np
import tensorflow as tf
import gym
import time

import pg.core as core

EPS = 1e-8


class GAEBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, ) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mu_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.log_std_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, done, val, logp, mu, log_std):
        """
        Store transition.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mu_buf[self.ptr] = mu
        self.log_std_buf[self.ptr] = log_std
        self.ptr += 1

    def finish_path(self, last_val=0.):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two" lines implement GAE-Lambda advantage calculation
        # deltas = rews[:-1] + self.gamma * \
        #     (1.0 - self.done_buf) * vals[1:] - vals[:-1]
        lastgaelam = 0.0
        lastret = rews[-1]
        for t in reversed(range(self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            delta = rews[t] + self.gamma * nondone * vals[t + 1] - vals[t]
            self.adv_buf[t] = lastgaelam = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            self.ret_buf[t] = lastret = rews[t] + \
                self.gamma * nondone * lastret

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / \
            (np.std(self.adv_buf) + EPS)
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf, self.mu_buf, self.log_std_buf]


def trpo(env_fn, actor_critic=core.mlp_actor_critic, seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, detla=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=10000, save_freq=10):

    seed = int(time.time()) % 1000
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    obs_ph = tf.placeholder(
        shape=(None, ) + obs_dim, dtype=tf.float32, name="obs_ph")
    act_ph = tf.placeholder(
        shape=(None, ) + act_dim, dtype=tf.float32, name="act_ph")
    adv_ph = tf.placeholder(
        shape=[None, ], dtype=tf.float32, name="adv_ph")
    ret_ph = tf.placeholder(
        shape=[None, ], dtype=tf.float32, name="ret_ph")
    logp_old_ph = tf.placeholder(
        shape=[None, ], dtype=tf.float32, name="logp_old_ph")

    # Main outputs from computation graph, plus placeholders for old pdist (for KL)
    pi, logp, logp_pi, mu, log_std, old_mu_ph, old_log_std_ph, d_kl, v = actor_critic(
        obs_ph, act_ph, action_space=env.action_space)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [obs_ph, act_ph, adv_ph, ret_ph,
               logp_old_ph, old_mu_ph, old_log_std_ph]

    get_action_ops = [pi, v, logp_pi, mu, log_std]

    # Experience buffer
    buf = GAEBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    print(
        f'\nNumber of parameters: \t pi: {var_counts[0]}, \t v: {var_counts[1]}')

    # TRPO losses
    ratio = tf.exp(logp - logp_old_ph)  # pi(a|s) / pi_old(a|s)
    pi_loss = -tf.reduce_mean(ratio * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Optimizer for value funtion
    train_vf = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    # Symbols needed for CG solver
    pi_params = core.get_vars('pi')
    pi_grads_flatted = core.flat_grad(pi_loss, pi_params)
    v_ph, hvp = core.hessian_vector_product(d_kl, pi_params)
    if damping_coeff > 0:
        hvp += damping_coeff * v_ph

    # Symbols for getting and setting params
    get_flatted_pi_params = core.flat_concat(pi_params)
    set_pi_params = core.assign_params_from_flat(v_ph, pi_params)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options,
                              log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    # sess.run(sync_all_params())

    # TODO: Model save

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        # Prepare hessian func, gradient eval
        inputs = {k: v for k, v in zip(all_phs, buf.get())}
        def Hx(x): return sess.run(hvp, feed_dict={**inputs, v_ph: x})
        g, pi_loss_old, v_loss_old = sess.run(
            [pi_grads_flatted, pi_loss, v_loss], feed_dict=inputs)

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = np.sqrt(2*detla/(np.dot(x, Hx(x))+EPS))
        old_params = sess.run(get_flatted_pi_params)

        for j in range(backtrack_iters):
            step = backtrack_coeff ** j
            sess.run(set_pi_params, feed_dict={
                     v_ph: old_params - alpha * x * step})
            kl, pi_loss_new = sess.run([d_kl, pi_loss], feed_dict=inputs)
            if kl < detla and pi_loss_new < pi_loss_old:
                print(f'Accepting new params at step {j} of line search.')
                break
            if j == backtrack_iters - 1:
                print('Line search failed! Keeping old params.')
                sess.run(set_pi_params, feed_dict={v_ph: old_params})

        for _ in range(train_v_iters):
            sess.run(train_vf, feed_dict=inputs)
        v_loss_new = sess.run(v_loss, feed_dict=inputs)

    start_time = time.time()
    obs = env.reset()
    ep_ret, ep_len = 0, 0

    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            agent_outs = sess.run(
                get_action_ops, feed_dict={obs_ph: obs.reshape(1, -1)})
            action, v_t, logp_pi_t, mu_t, log_std_t = \
                agent_outs[0][0], agent_outs[1], agent_outs[2], agent_outs[3], agent_outs[4]

            next_obs, reward, done, _ = env.step(action)
            ep_ret += reward
            ep_len += 1

            buf.store(obs, action, reward, done, v_t,
                      logp_pi_t, mu_t, log_std_t)

            obs = next_obs

            terminal = done or (ep_len >= max_ep_len)
            if terminal:
                # Summary
                print(f'ep_len: {ep_len}, ep_ret: {ep_ret}')
                obs = env.reset()
                ep_ret, ep_len = 0, 0

            if t >= steps_per_epoch - 1:
                if not terminal:
                    print(
                        f'Warning: trajectory cut off by epoch at {ep_len} steps.')
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if done else \
                    sess.run(v, feed_dict={obs_ph: obs.reshape(1, - 1)})
                buf.finish_path(last_val)

        # Save model
        if epoch % save_freq == 0 or epoch == epochs - 1:
            pass

        update()

        # Summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    trpo(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
         gamma=args.gamma,
         steps_per_epoch=args.steps, epochs=args.epochs)
