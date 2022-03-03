import numpy as np
import tensorflow as tf
import gym
import time
import os
import json

from lib.utils.json_tools import json_serializable

EPS = 1e-8


def gaussian_likelihood(x, mu, logstd):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(logstd) + EPS)) ** 2 +
                      2 * logstd + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, ) + obs_dim, dtype=np.float32)
        self.act_buf = np.zeros((size, ) + act_dim, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size

    def store(self, obs, act, rew, done, val, logp):
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
        self.ptr += 1

    def finish_path(self, last_val=0.):
        assert self.ptr == self.max_size
        rews = np.append(self.rew_buf, last_val)
        vals = np.append(self.val_buf, last_val)

        # the next two" lines implement GAE-Lambda advantage calculation
        lastgaelam = 0.0
        lastret = rews[-1]
        for t in reversed(range(self.ptr)):
            nondone = 1.0 - self.done_buf[t]
            delta = rews[t] + self.gamma * nondone * vals[t + 1] - vals[t]
            self.adv_buf[t] = lastgaelam = delta + \
                self.gamma * self.lam * nondone * lastgaelam
            self.ret_buf[t] = lastret = rews[t] + \
                self.gamma * nondone * lastret

    def get(self):
        assert self.ptr == self.max_size
        self.ptr = 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / \
            (np.std(self.adv_buf) + EPS)
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf]


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_dim, name=None):
        super(ActorMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.tanh
        kernel_initializer = tf.initializers.orthogonal
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            ac_dim[0],
            kernel_initializer=kernel_initializer, name='fc3')

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
        kernel_initializer = tf.initializers.orthogonal
        self.dense1 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            64, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            1, kernel_initializer=kernel_initializer, name='fc3')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class PPOAgent():
    def __init__(self, sess, obs_dim, act_dim, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, target_kl=0.01):
        self.sess = sess
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

        self._build_network()
        [self.obs_ph, self.all_phs, self.get_action_ops,
         self.v, self.pi_loss, self.v_loss,
         self.approx_kl, self.approx_ent, self.clipfrac,
         self.train_pi, self.train_v] = self._build_train_op()
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
        obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        act_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.act_dim, dtype=tf.float32, name="act_ph")
        adv_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="adv_ph")
        ret_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="ret_ph")
        logp_old_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="logp_old_ph")

        # Probability distribution
        logits = self.actor(obs_ph)
        logstd = tf.compat.v1.get_variable(
            name='pi/logstd',
            initializer=-0.5 * np.ones(self.act_dim, dtype=np.float32))
        std = tf.exp(logstd)
        pi = logits + tf.random.normal(tf.shape(logits)) * std
        logp_a = gaussian_likelihood(act_ph, logits, logstd)
        logp_pi = gaussian_likelihood(pi, logits, logstd)

        # State value
        v = tf.compat.v1.squeeze(self.critic(obs_ph), axis=1)

        get_action_ops = [pi, v, logp_pi]

        all_phs = [obs_ph, act_ph, adv_ph, ret_ph, logp_old_ph]

        # PPO objectives
        ratio = tf.exp(logp_a - logp_old_ph)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.compat.v1.where(adv_ph > 0,
                                     (1 + self.clip_ratio) * adv_ph,
                                     (1 - self.clip_ratio) * adv_ph)
        pi_loss = -tf.reduce_mean(
            tf.minimum(ratio * adv_ph, min_adv))
        v_loss = tf.reduce_mean((ret_ph - v) ** 2)

        # Info (useful to watch during learning)
        # a sample estimate for KL-divergence, easy to compute
        approx_kl = tf.reduce_mean(logp_old_ph - logp_a)
        # a sample estimate for entropy, also easy to compute
        approx_ent = tf.reduce_mean(-logp_a)
        clipped = tf.logical_or(
            ratio > (1 + self.clip_ratio), ratio < (1 - self.clip_ratio))
        clipfrac = tf.reduce_mean(
            tf.cast(clipped, tf.float32))

        # Optimizers
        train_pi = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.pi_lr).minimize(pi_loss)
        train_v = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.vf_lr).minimize(v_loss)

        return [obs_ph, all_phs, get_action_ops,
                v, pi_loss, v_loss,
                approx_kl, approx_ent, clipfrac,
                train_pi, train_v]

    def update(self, buf_data):
        inputs = {k: v for k, v in zip(self.all_phs, buf_data)}

        pi_loss_old, v_loss_old, ent = self.sess.run(
            [self.pi_loss, self.v_loss, self.approx_ent], feed_dict=inputs)

        for i in range(self.train_pi_iters):
            _, kl = self.sess.run(
                [self.train_pi, self.approx_kl], feed_dict=inputs)
            if kl > 1.5 * self.target_kl:
                print(f'Early stopping at step {i} due to reaching max kl.')
                break
        for _ in range(self.train_v_iters):
            self.sess.run(self.train_v, feed_dict=inputs)

        pi_loss_new, v_loss_new, kl, cf = self.sess.run(
            [self.pi_loss, self.v_loss, self.approx_kl, self.clipfrac],
            feed_dict=inputs)

    def _build_saver(self):
        pi_params = self._get_var_list('pi')
        vf_params = self._get_var_list('vf')
        return tf.compat.v1.train.Saver(var_list=pi_params + vf_params,
                                        max_to_keep=4)

    def select_action(self, obs):
        return self.sess.run(
            self.get_action_ops, feed_dict={self.obs_ph: obs.reshape(1, -1)})

    def compute_v(self, obs):
        return self.sess.run(
            self.v, feed_dict={self.obs_ph: obs.reshape(1, -1)})

    def bundle(self, checkpoint_dir, iteration):
        if not os.path.exists(checkpoint_dir):
            raise
        self.actor.save_weights(
            os.path.join(checkpoint_dir, 'best_model_actor.h5'), save_format='h5')
        self.critic.save_weights(
            os.path.join(checkpoint_dir, 'best_model_critic.h5'), save_format='h5')
        self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, 'tf_ckpt'),
            global_step=iteration)

    def unbundle(self, checkpoint_dir, iteration=None):
        if not os.path.exists(checkpoint_dir):
            raise
        # Load the best weights without iteraion.
        if iteration is None:
            self.actor.load_weights(
                os.path.join(checkpoint_dir, 'best_model_actor.h5'))
            self.critic.load_weights(
                os.path.join(checkpoint_dir, 'best_model_critic.h5'))
        else:
            self.saver.restore(
                self.sess,
                os.path.join(checkpoint_dir, f'tf_ckpt-{iteration}'))


def evaluate(env_eval, agent, eval_ep_n=10, max_ep_len=10000):
    ret_sum = 0.
    len_sum = 0

    for i in range(1, eval_ep_n + 1):
        obs = env_eval.reset()
        ep_ret = 0.
        ep_len = 0
        while True:
            pi = agent.select_action(obs)[0]
            ac = pi[0]
            obs, reward, done, _ = env_eval.step(ac)
            ep_ret += reward
            ep_len += 1
            if done or ep_len >= max_ep_len:
                print(f'\reval: {i}/{eval_ep_n}', end='')
                ret_sum += ep_ret
                len_sum += ep_len
                break
    avg_eval_ret = ret_sum / eval_ep_n
    avg_eval_len = len_sum / eval_ep_n
    print(f'\navg_eval_len: {ep_len}, avg_eval_ret: {ep_ret: .1f}')
    return avg_eval_ret, avg_eval_len


def ppo(base_dir, env_name, total_steps=int(1e6), horizon=2048,
        gamma=0.99, lam=0.95, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, target_kl=0.01,
        max_ep_len=10000, eval_freq=10, allow_eval=True):

    # Create dir
    summary_dir = os.path.join(base_dir, "tf1_summary")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    progress_txt = os.path.join(base_dir, 'progress.txt')
    with open(progress_txt, 'w') as f:
        f.write('Step\tValue\n')

    # Tensorboard
    summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    seed = int(time.time()) % 1000
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    env.seed(seed)
    env_eval = gym.make(env_name)
    env_eval.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Experience buffer
    buf = PPOBuffer(obs_dim, act_dim, horizon, gamma, lam)

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                        log_device_placement=False))

    agent = PPOAgent(sess, obs_dim, act_dim)

    sess.run(tf.compat.v1.global_variables_initializer())

    start_time = time.time()
    obs = env.reset()
    ep_ret, ep_len = 0, 0
    ep_count = 0
    max_ep_ret = 0

    epochs = total_steps // horizon
    for epoch in range(1, epochs + 1):
        for t in range(1, horizon + 1):
            [pi, v_t, logp_pi_t] = agent.select_action(obs)
            ac = pi[0]

            next_obs, reward, done, _ = env.step(ac)
            ep_ret += reward
            ep_len += 1

            buf.store(obs, ac, reward, done, v_t, logp_pi_t)

            obs = next_obs

            if done or (ep_len >= max_ep_len):
                # Episode summary
                episode_summary = tf.compat.v1.Summary(value=[
                    tf.compat.v1.Summary.Value(
                        tag="episode_info/reward", simple_value=ep_ret),
                    tf.compat.v1.Summary.Value(
                        tag="episode_info/length", simple_value=ep_len)
                ])
                summary_writer.add_summary(episode_summary, ep_count)
                print(f'Epoch: {epoch}/{epochs}, '
                      f'ep_len: {ep_len}, ep_ret: {ep_ret:.1f}, '
                      f'train: {epoch/epochs:.1%}')
                # Episode restart
                obs = env.reset()
                ep_ret, ep_len = 0, 0
                ep_count += 1

            if t == horizon:
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0.0 if done else agent.compute_v(obs)
                buf.finish_path(last_val)
                break

        buf_data = buf.get()
        agent.update(buf_data)

        # Evaluate
        step = epoch * horizon
        if epoch % eval_freq == 0 and allow_eval:
            avg_ret, evg_len = evaluate(
                env_eval, agent, eval_ep_n=10, max_ep_len=max_ep_len)
            # Summary
            eval_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag='eval/avg_len', simple_value=evg_len),
                tf.compat.v1.Summary.Value(
                    tag='eval/avg_reward', simple_value=avg_ret)
            ])
            summary_writer.add_summary(eval_summary, step)
            # Save the best weights
            if avg_ret >= max_ep_ret:
                print(f'Saving weights into {checkpoint_dir}')
                agent.bundle(checkpoint_dir, epoch)
                max_ep_ret = avg_ret
            # Log data
            with open(progress_txt, 'a') as f:
                f.write(f"{step}\t{avg_ret}\n")
    print(f"Results saved into {base_dir}")
    summary_writer.flush()
    env.close()
    env_eval.close()


def main(args):
    if not os.path.exists(args.disk_dir):
        raise
    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    dir_name = args.dir_name or (args.exp_name + '-' + timestamp)
    exp_name = args.exp_name
    env_name = args.env_name
    base_dir = os.path.join(args.disk_dir, f"my_results/{env_name}/{dir_name}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    config = json_serializable(locals())
    # Save config_json
    config_json = json.dumps(config, sort_keys=False,
                             indent=4, separators=(',', ': '))
    with open(os.path.join(base_dir, "config.json"), 'w') as out:
        out.write(config_json)
    # Run
    allow_eval = not args.noneval
    ppo(base_dir=base_dir, env_name=env_name, allow_eval=allow_eval)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default=None, help='Dir name')
    parser.add_argument('--disk_dir', type=str,
                        default='/data/hanjl', help='Data disk dir')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--noneval', action='store_true', help='No eval')
    args = parser.parse_args()

    main(args)
