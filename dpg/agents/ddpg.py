import numpy as np
import tensorflow as tf
import os

import dpg.buffer.replaybuffer as Buffer


class ActorMLP(tf.keras.Model):
    def __init__(self, ac_dim, max_ac, name):
        super(ActorMLP, self).__init__(name=name)
        self.ac_limit = max_ac
        activation_fn = tf.keras.activations.relu
        output_activation_fn = tf.keras.activations.tanh
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            ac_dim[0], activation=output_activation_fn,
            kernel_initializer=kernel_initializer, name='fc3')

    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x * self.ac_limit


class CriticMLP(tf.keras.Model):
    def __init__(self, name):
        super(CriticMLP, self).__init__(name=name)
        activation_fn = tf.keras.activations.relu
        kernel_initializer = None
        self.dense1 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc1')
        self.dense2 = tf.keras.layers.Dense(
            256, activation=activation_fn,
            kernel_initializer=kernel_initializer, name='fc2')
        self.dense3 = tf.keras.layers.Dense(
            1, activation=None,
            kernel_initializer=kernel_initializer, name='fc3')

    def call(self, state, action):
        x = tf.cast(tf.concat([state, action], axis=-1), tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return tf.squeeze(x, axis=1)


class DDPGAgent():
    def __init__(self, sess, obs_dim, ac_dim, ac_limit,
                 mem_capacity=int(1e6), batch_size=256,
                 gamma=0.99, actor_lr=3e-4, critic_lr=3e-4,
                 tau=0.005, noise_scale=0.1):
        self.sess = sess
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.ac_limit = ac_limit
        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.noise_scale = noise_scale

        self.buffer = Buffer.ReplayBuffer(obs_dim, ac_dim, mem_capacity)
        self._build_network()
        self._build_train_op()

    # Note: Required to be called after _build_train_op(), otherwise return []
    def _get_var_list(self, name='online'):
        scope = tf.compat.v1.get_default_graph().get_name_scope()
        vars = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            scope=os.path.join(scope, name))
        return vars

    def _build_network(self):
        self.actor = ActorMLP(self.ac_dim, self.ac_limit, 'online/actor')
        self.critic = CriticMLP('online/critic')
        self.actor_target = ActorMLP(
            self.ac_dim, self.ac_limit, 'target/actor')
        self.critic_target = CriticMLP('target/critic')

    def _build_train_op(self):
        self.obs_ph = obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="obs_ph")
        self.next_obs_ph = next_obs_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.obs_dim, dtype=tf.float32, name="next_obs_ph")
        self.ac_ph = ac_ph = tf.compat.v1.placeholder(
            shape=(None, ) + self.ac_dim, dtype=tf.float32, name="ac_ph")
        self.rew_ph = rew_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="rew_ph")
        self.done_ph = done_ph = tf.compat.v1.placeholder(
            shape=[None, ], dtype=tf.float32, name="done_ph")

        pi = self.actor(obs_ph)
        actor_loss = tf.reduce_mean(-self.critic(obs_ph, pi))

        target_q_next = self.critic_target(
            next_obs_ph, self.actor_target(next_obs_ph))
        target = rew_ph + (1 - done_ph) * self.gamma * target_q_next
        target_nograd = tf.stop_gradient(target)

        q_value = self.critic(obs_ph, ac_ph)

        critic_loss = tf.reduce_mean((q_value - target_nograd) ** 2)

        actor_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.actor_lr)
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.critic_lr)

        train_actor_op = actor_optimizer.minimize(
            actor_loss, var_list=self._get_var_list('online/actor'))
        train_critic_op = critic_optimizer.minimize(
            critic_loss, var_list=self._get_var_list('online/critic'))

        trainable_online = (self._get_var_list('online/actor')
                            + self._get_var_list('online/critic'))
        trainable_target = (self._get_var_list('target/actor')
                            + self._get_var_list('target/critic'))

        target_update = []
        for (w_online, w_target) in zip(trainable_online, trainable_target):
            w_tau = self.tau * w_online + (1 - self.tau) * w_target
            target_update.append(w_target.assign(w_tau, use_locking=True))

        target_init = []
        for (w_online, w_target) in zip(trainable_online, trainable_target):
            target_init.append(w_target.assign(w_online, use_locking=True))

        self.pi = pi
        self.target_update = target_update
        self.target_init = target_init
        self.train_actor_op = train_actor_op
        self.train_critic_op = train_critic_op

        meanq = tf.reduce_mean(q_value, axis=-1)
        self.stats_list = [actor_loss, critic_loss, meanq]
        self.loss_names = ['actor_loss', 'critic_loss', 'q_value']
        assert len(self.stats_list) == len(self.loss_names)

    def select_action(self, obs, noise=True):
        pi = self.sess.run(
            self.pi, feed_dict={self.obs_ph: obs.reshape(1, -1)})
        ac = pi[0]
        if noise:
            ac_noise = np.random.normal(
                0, self.ac_limit * self.noise_scale, size=self.ac_dim)
            ac = np.clip(ac + ac_noise, -self.ac_limit, self.ac_limit)
        return ac

    def target_params_init(self, init=False):
        self.sess.run(self.target_init)

    def update(self, logger):
        buf_data = self.buffer.sample_batch(self.batch_size)
        [obs, next_obs, ac, rew, done] = buf_data

        inputs = {
            self.obs_ph: obs,
            self.next_obs_ph: next_obs,
            self.ac_ph: ac,
            self.rew_ph: rew,
            self.done_ph: done
        }

        losses = self.sess.run(
            self.stats_list + [self.train_actor_op, self.train_critic_op], feed_dict=inputs)[:-2]
        for (lossval, lossname) in zip(losses, self.loss_names):
            logger.logkv('loss/' + lossname, lossval)
        logger.logkv("loss/actor_lr", self.actor_lr)
        logger.logkv("loss/critic_lr", self.critic_lr)
        self.sess.run(self.target_update)

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
