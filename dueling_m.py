import gym
import numpy as np
import os
import random
import tensorflow as tf
import time
import json
from argparse import ArgumentParser

from lib.dqn_utils_m import *

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
	"""Returns the current epsilon for the agent's epsilon-greedy policy.
	This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
	al., 2015). The schedule is as follows:
		Begin at 1. until warmup_steps steps have been taken; then
		Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
		Use epsilon from there on.
	Args:
		decay_period: float, the period over which epsilon is decayed.
		step: int, the number of training steps completed so far.
		warmup_steps: int, the number of steps taken before epsilon is decayed.
		epsilon: float, the final value to which to decay the epsilon parameter.
	Returns:
		A float, the current epsilon value computed according to the schedule.
	"""
	steps_left = decay_period + warmup_steps - step
	bonus = (1.0 - epsilon) * steps_left / decay_period
	bonus = np.clip(bonus, 0., 1. - epsilon)
	return epsilon + bonus

class DuelingNetwork(tf.keras.Model):
	"""The convolutional network used to compute the agent's Q-values."""

	def __init__(self, num_actions, name=None):
		"""
		Creates the layers used for calculating Q-values.
		Args:
		num_actions: int, number of actions.
		name: str, used to create scope for network parameters.
		"""
		super(DuelingNetwork, self).__init__(name=name)

		self.num_actions = num_actions
		# Defining layers.
		activation_fn = tf.keras.activations.relu
		# Setting names of the layers manually to make variable names more similar
		# with tf.slim variable names/checkpoints.
		self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
			  data_format='channels_first', activation=activation_fn, name='conv1')
		self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
			  data_format='channels_first', activation=activation_fn, name='conv2')
		self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
			  data_format='channels_first', activation=activation_fn, name='conv3')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
											name='fc1')
		self.dense2 = tf.keras.layers.Dense(self.num_actions, name='fc2')
		self.dense3 = tf.keras.layers.Dense(512, activation=activation_fn,
											name='fc3')
		self.dense4 = tf.keras.layers.Dense(1, name='fc4')

	def call(self, state):
		"""Creates the output tensor/op given the state tensor as input.
		See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
		information on this. Note that tf.keras.Model implements `call` which is
		wrapped by `__call__` function by tf.keras.Model.
		Parameters created here will have scope according to the `name` argument
		given at `.__init__()` call.
		Args:
		state: Tensor, input tensor.
		Returns:
		collections.namedtuple, output ops (graph mode) or output tensors (eager).
		"""
		x = tf.cast(state, tf.float32)
		x = x / 255
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.flatten(x)
		advantage = self.dense1(x)
		advantage = self.dense2(advantage)
		value = self.dense3(x)
		value = self.dense4(value)
		x = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
		return x

class Agent():
	def __init__(self,
				 sess,
				 num_actions,
				 replay_capacity=1000000,
				 replay_min_size=50000,
				 update_period=4,
				 target_update_period=10000,
				 epsilon_train=0.01,
				 epsilon_eval=0.001,
				 epsilon_decay_period=1000000,
				 gamma=0.99,
				 batch_size=32,
				 eval_mode=False,
				 max_tf_checkpoints_to_keep=4,
				 # https://www.tensorflow.org/api_docs/python/tf/compat\
				 # /v1/train/RMSPropOptimizer#migrate-to-tf2_2
				 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025, \
													 decay=0.95, \
													 epsilon=1e-6, \
													 centered=False),
                 summary_writer=None,
                 summary_writing_frequency=500):
		self.config = json_serializable(locals())
		self.config['optimizer'] = optimizer.get_name()
		self._sess = sess
		self.num_actions = num_actions
		self.replay_min_size = replay_min_size
		self.update_period = update_period
		self.target_update_period = target_update_period
		self.epsilon_train = epsilon_train
		self.epsilon_eval = epsilon_eval
		self.epsilon_decay_period = epsilon_decay_period
		self.gamma = gamma
		self.eval_mode = eval_mode
		self.max_tf_checkpoints_to_keep = max_tf_checkpoints_to_keep
		self.optimizer = optimizer
		self.summary_writer = summary_writer
		self.summary_writing_frequency = summary_writing_frequency
		self._replay = ReplayMemory(memory_size=replay_capacity, batch_size=batch_size)
		self._history = History()
		self._build_networks()
		self._train_op = self._build_train_op()
		self._sync_qt_ops = self._build_sync_op()
		self._state_processer = self._build_state_processer()
		self._saver = self._build_saver()

		if self.summary_writer is not None:
			# All tf.summaries should have been defined prior to running this.
			self._merged_summaries = tf.summary.merge_all()

		self.training_steps = 0

	def _build_networks(self):
		self.online_network = DuelingNetwork(num_actions=self.num_actions, name="online")
		self.target_network = DuelingNetwork(num_actions=self.num_actions, name="target")

	# Note: Required to be called after _build_train_op(), otherwise return []
	def _get_var_list(self, name='online'):
		scope = tf.get_default_graph().get_name_scope()
		trainable_variables = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES,
			scope=os.path.join(scope, name))
		return trainable_variables

	def _build_train_op(self):
		self.state_ph = tf.placeholder(shape=[None, 4, 84, 84], dtype=tf.uint8, name="state_ph")
		self.next_state_ph = tf.placeholder(shape=[None, 4, 84, 84], dtype=tf.uint8, name="next_state_ph")
		self.actions_ph = tf.placeholder(shape=[None,], dtype=tf.int32, name="actions_ph")
		self.rewards_ph = tf.placeholder(shape=[None,], dtype=tf.float32, name="rewards_ph")
		self.terminals_ph = tf.placeholder(shape=[None,], dtype=tf.float32, name="terminals_ph")

		self.online_net_output = self.online_network(self.state_ph) # (32, 4)
		self.target_net_output = self.target_network(self.next_state_ph) # (32, 4)
		self.q_argmax = tf.argmax(self.online_net_output, axis=1)[0] # (1, ) => ()

		next_q_max = tf.reduce_max(self.target_net_output, axis=1) # (32, 4) => (32,)
		target = self.rewards_ph + self.gamma * (1 - self.terminals_ph) * next_q_max # (32,)
		target_nograd = tf.stop_gradient(target)

		q_value_chosen = tf.squeeze( \
						     tf.gather(self.online_net_output, \
							 	tf.expand_dims(self.actions_ph, axis=-1), axis=1, batch_dims=1)) # (32,)

		losses = tf.losses.huber_loss(
			target_nograd, q_value_chosen, reduction=tf.losses.Reduction.NONE)
		loss = tf.reduce_mean(losses)
		if self.summary_writer is not None:
			with tf.variable_scope('losses'):
				tf.summary.scalar(name='huberloss', tensor=loss)
			with tf.variable_scope('q_estimate'):
				tf.summary.scalar(name='max_q_value', tensor=tf.reduce_max(self.online_net_output))
				tf.summary.scalar(name='avg_q_value', tensor=tf.reduce_mean(self.online_net_output))
		return self.optimizer.minimize(loss, var_list=self._get_var_list())

	def _build_state_processer(self):
		self.raw_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
		self.output = tf.image.rgb_to_grayscale(self.raw_state)
		# self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
		self.output = tf.image.resize(
				self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		return tf.squeeze(self.output)

	def _build_sync_op(self):
		"""Builds ops for assigning weights from online to target network.
		Returns:
		ops: A list of ops assigning weights from online to target network.
		"""
		# Get trainable variables from online and target DQNs
		sync_qt_ops = []
		trainable_online = self._get_var_list()
		trainable_target = self._get_var_list('target')

		for (w_online, w_target) in zip(trainable_online, trainable_target):
			sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
		return sync_qt_ops
	
	def _build_saver(self):
		return tf.train.Saver(var_list=self._get_var_list(), \
				max_to_keep=self.max_tf_checkpoints_to_keep)

	def bundle(self):
		self._saver.save(
			self._sess,
			os.path.join(self.checkpoint_dir, 'tf_ckpt'),
			global_step=self.training_steps)

	def select_action(self):
		if self.eval_mode:
			epsilon = self.epsilon_eval
		else:
			epsilon = linearly_decaying_epsilon(
				decay_period=self.epsilon_decay_period,
				step=self.training_steps,
				warmup_steps=self.replay_min_size,
				epsilon=self.epsilon_train)
		if random.random() <= epsilon:
			return random.randint(0, self.num_actions - 1)
		else:
			return self._sess.run(self.q_argmax, {self.state_ph: self._history.get()})

	def step(self, action, observation, reward, terminal):
		self._history.add(observation)
		# If eval, store and train are no longer needed.
		if self.eval_mode:
			return
		# Store transition
		self._replay.add(action, observation, reward, terminal)
		# Train
		if self._replay.count > self.replay_min_size:
			if self.training_steps % self.update_period == 0:
				states, actions, rewards, next_states, terminals = \
					self._replay.sample()
				feed_fict = {self.state_ph: states, \
							 self.next_state_ph: next_states, \
							 self.actions_ph: actions, \
							 self.rewards_ph: rewards, \
							 self.terminals_ph: terminals}
				self._sess.run(self._train_op, feed_fict)

				if self.summary_writer is not None and \
					self.training_steps % self.summary_writing_frequency == 0:
					summary = self._sess.run(self._merged_summaries, feed_fict)
					self.summary_writer.add_summary(summary, self.training_steps)

			if self.training_steps % self.target_update_period == 0:
				self._sess.run(self._sync_qt_ops)

		self.training_steps += 1

	def begin_episode(self, observation):
		for _ in range(4):
			self._history.add(observation)

class Runner():
	def __init__(self,
				 base_dir,
				 env_name,
				 num_iterations=200,
				 min_train_steps=60000,
				 evaluation_steps=20000,
				 max_steps_per_episode=27000,
				 clip_rewards=True):
		if not os.path.join(base_dir):
			raise
		self.config = json_serializable(locals())
		self._base_dir = base_dir
		self._num_iterations = num_iterations
		self._min_train_steps = min_train_steps
		self._evaluation_steps = evaluation_steps
		self._max_steps_per_episode = max_steps_per_episode
		self._clip_rewards = clip_rewards
		# env
		self._env = create_atari_environment(env_name)
		num_actions = self._env.action_space.n
		# summary_writer
		summary_dir = os.path.join(self._base_dir, "tf1_summary")
		if not os.path.exists(summary_dir):
			os.makedirs(summary_dir)
		self._summary_writer = tf.summary.FileWriter(summary_dir)
		# sess
		gpu_options = tf.GPUOptions(allow_growth=True)
		self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		# agent
		self._agent = Agent(sess=self._sess, num_actions=num_actions, \
			 				summary_writer=self._summary_writer)
		self._summary_writer.add_graph(graph=tf.get_default_graph())
		self._sess.run(tf.global_variables_initializer())

		self.config['env_config'] = self._env.config
		self.config['agent_config'] = self._agent.config

		self.iteration = 0
		self.total_episode = 0

	def _run_one_episode(self):
		episode_length = 0
		episode_reward = 0.

		observation = self._env.reset()
		self._agent.begin_episode(observation)
		while True:
			print(f"\rlength: {episode_length}, reward: {episode_reward}", end='')

			action = self._agent.select_action()
			observation, reward, terminal, _ = self._env.step(action)

			episode_reward += reward
			episode_length += 1

			reward_clip = np.clip(reward, -1, 1)

			if self._env.game_over or (episode_length >= self._max_steps_per_episode):
				# we lose all lifes
				self._agent.step(action, observation, reward_clip, True)
				break
			elif terminal:
				# If we lose a life but the episode is not over
				# Terminal on life loss = True
				self._agent.step(action, observation, reward_clip, True)
				self._agent.begin_episode(observation)
			else:
				self._agent.step(action, observation, reward_clip, False)
		return episode_length, episode_reward

	def _run_one_phase(self, min_steps, eval_mode=False):
		step_count = 0
		num_episodes = 0
		sum_rewards = 0.

		self._agent.eval_mode = eval_mode
		start_time = time.time()

		while step_count < min_steps:
			print(f"\n@iter: {self.iteration}/{self._num_iterations}, train: {not eval_mode}, " \
				  f"step: {step_count}/{min_steps}, total_ep: {self.total_episode}")
			episode_length, episode_reward = self._run_one_episode()
			step_count += episode_length
			sum_rewards += episode_reward
			num_episodes += 1
			# episode_info
			if not eval_mode:
				episode_summary = tf.Summary(value=[
					tf.Summary.Value(simple_value=episode_reward, tag="episode_info/reward"),
					tf.Summary.Value(simple_value=episode_length, tag="episode_info/length")
				])
				self._summary_writer.add_summary(episode_summary, self.total_episode)
				self.total_episode += 1
		time_delta = time.time() - start_time
		average_steps_per_second = step_count / time_delta
		average_rewrd = sum_rewards / num_episodes
		return num_episodes, average_rewrd, average_steps_per_second

	def _save_tensorboard_summaries(self,
									num_episodes_train,
									average_reward_train,
									average_steps_per_second,
									num_episodes_eval,
									average_reward_eval):
		summary = tf.Summary(value=[
			tf.Summary.Value(
				tag='train/num_episodes', simple_value=num_episodes_train),
			tf.Summary.Value(
				tag='train/average_reward', simple_value=average_reward_train),
			tf.Summary.Value(
				tag='train/average_steps_per_second',
				simple_value=average_steps_per_second),
			tf.Summary.Value(
				tag='eval/num_episodes', simple_value=num_episodes_eval),
			tf.Summary.Value(
				tag='eval/average_reward', simple_value=average_reward_eval)
		])
		self._summary_writer.add_summary(summary, self.iteration)

	def _checkpoint_experiment(self):
		checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self._agent._saver.save(
			self._sess,
			os.path.join(checkpoint_dir, 'tf_ckpt'),
			global_step=self.iteration)

	def _log_experiment(self):
		pass

	def run_experiment(self):
		for self.iteration in range(self._num_iterations):
			num_episodes_train, average_reward_train, average_steps_per_second = \
				self._run_one_phase(min_steps=self._min_train_steps)
			num_episodes_eval, average_reward_eval, _ = \
				self._run_one_phase(min_steps=self._evaluation_steps, eval_mode=True)
			self._save_tensorboard_summaries(num_episodes_train, \
											 average_reward_train, \
											 average_steps_per_second, \
											 num_episodes_eval, \
											 average_reward_eval)
			self._checkpoint_experiment()
			self._log_experiment()
		print(f"\nResults have been saved into {self._base_dir}")
		self._summary_writer.flush()
		self._env.close()

def main(args):
	if not os.path.exists(args.disk_dir):
		raise
	timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
	exp_name = args.exp_name or (args.tag + '-' + timestamp)
	env_name = '{}NoFrameskip-v0'.format(args.env_name)
	base_dir = os.path.join(args.disk_dir, f"my_results/{env_name}/{exp_name}")
	if not os.path.exists(base_dir):
		os.makedirs(base_dir)
	config = json_serializable(locals())
	# Runner
	runner = Runner(base_dir=base_dir, env_name=env_name)
	config['runner_config'] = runner.config
	# Save config_json
	config_json = json.dumps(config, sort_keys=False, indent=4, separators=(',', ': '))
	with open(os.path.join(base_dir, "config.json"), 'w') as out:
		out.write(config_json)
	# Run
	runner.run_experiment()

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--tag', type=str, default='.debug', help='Used as part exp_name')
	parser.add_argument('--exp_name', type=str, default=None, help='Used as full exp_name')
	parser.add_argument('--env_name', type=str, default='Breakout', help='Env name')
	parser.add_argument('--disk_dir', type=str, default='/data/hanjl', help='Data disk dir')
	args, unknown_args = parser.parse_known_args()
	main(args)