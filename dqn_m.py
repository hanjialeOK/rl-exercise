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

class NatureDQNNetwork(tf.keras.Model):
	"""The convolutional network used to compute the agent's Q-values."""

	def __init__(self, num_actions, name=None):
		"""
		Creates the layers used for calculating Q-values.
		Args:
		num_actions: int, number of actions.
		name: str, used to create scope for network parameters.
		"""
		super(NatureDQNNetwork, self).__init__(name=name)

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
		x = self.dense1(x)
		x = self.dense2(x)
		return x

class Agent():
	def __init__(self,
				 sess,
				 num_actions,
				 exp_dir=None,
				 replay_capacity=1000000,
				 replay_min_size=50000,
				 update_period=4,
				 target_update_period=10000,
				 epsilon_end=0.01,
				 epsilon_decay_period=1000000,
				 gamma=0.99,
				 batch_size=32,
				 max_tf_checkpoints_to_keep=4,
				 # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/RMSPropOptimizer
				 # #migrate-to-tf2_2
				 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025, \
													 decay=0.95, \
													 epsilon=1e-6, \
													 centered=False),
                 summary_writer=None,
                 summary_writing_frequency=500):
		self.config = {}
		self.config['num_actions'] = num_actions
		self.config['replay_capacity'] = replay_capacity
		self.config['replay_min_size'] = replay_min_size
		self.config['update_period'] = update_period
		self.config['target_update_period'] = target_update_period
		self.config['epsilon_end'] = epsilon_end
		self.config['gamma'] = gamma
		self.config['batch_size'] = batch_size
		self.config['max_tf_checkpoints_to_keep'] = max_tf_checkpoints_to_keep
		self.config['optimizer'] = optimizer.get_name()
		self.config['summary_writing_frequency'] = summary_writing_frequency
		self.sess = sess
		self.num_actions = num_actions
		self.exp_dir = exp_dir
		self.replay_min_size = replay_min_size
		self.update_period = update_period
		self.target_update_period = target_update_period
		self.epsilon_end = epsilon_end
		self.epsilon_decay_period = epsilon_decay_period
		self.gamma = gamma
		self.max_tf_checkpoints_to_keep = max_tf_checkpoints_to_keep
		self.optimizer = optimizer
		self.summary_writer = summary_writer
		self.summary_writing_frequency = summary_writing_frequency
		self.training_steps = 0
		self.replay_memory = ReplayMemory(memory_size=replay_capacity, batch_size=batch_size)
		self.history = History()
		self._build_networks()
		self._train_op = self._build_train_op()
		self._sync_qt_ops = self._build_sync_op()
		self._state_processer = self._build_state_processer()
		self._saver = self._build_saver()

		if self.summary_writer is not None:
			# All tf.summaries should have been defined prior to running this.
			self._merged_summaries = tf.summary.merge_all()

	def _build_networks(self):
		self.online_network = NatureDQNNetwork(num_actions=self.num_actions, name="online")
		self.target_network = NatureDQNNetwork(num_actions=self.num_actions, name="target")
	
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
		self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
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
		self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		return tf.train.Saver(var_list=self._get_var_list(), \
				max_to_keep=self.max_tf_checkpoints_to_keep)

	def select_action(self):
		epsilon = linearly_decaying_epsilon(
			decay_period=self.epsilon_decay_period,
			step=self.training_steps,
			warmup_steps=self.replay_min_size,
			epsilon=self.epsilon_end
		)
		if random.random() <= epsilon:
			return random.randint(0, self.num_actions - 1)
		else:
			return self.sess.run(self.q_argmax, {self.state_ph: self.history.get()})

	def step(self, action, observation, reward, terminal):
		state = self.sess.run(self._state_processer, {self.raw_state: observation})
		self.history.add(state)
		self.replay_memory.add(action, state, reward, terminal)

		if self.replay_memory.count > self.replay_min_size:
			if self.training_steps % self.update_period == 0:
				states, actions, rewards, next_states, terminals = \
					self.replay_memory.sample()
				feed_fict = {self.state_ph: states, \
							 self.next_state_ph: next_states, \
							 self.actions_ph: actions, \
							 self.rewards_ph: rewards, \
							 self.terminals_ph: terminals}
				self.sess.run(self._train_op, feed_fict)

				if self.summary_writer is not None and \
					self.training_steps % self.summary_writing_frequency == 0:
					summary = self.sess.run(self._merged_summaries, feed_fict)
					self.summary_writer.add_summary(summary, self.training_steps)

			if self.training_steps % self.target_update_period == 0:
				self.sess.run(self._sync_qt_ops)

			if (self.training_steps % 1000000 == 999999):
				self._saver.save(
					self.sess,
					os.path.join(self.checkpoint_dir, 'tf_ckpt'),
					global_step=self.training_steps)

		self.training_steps += 1

	def reset_history(self, observation):
		state = self.sess.run(self._state_processer, feed_dict={self.raw_state: observation})
		for _ in range(4):
			self.history.add(state)


def main(args):
	name = args.name
	tag = args.tag
	num_episodes = args.num_ep
	# Create environment
	env = gym.make('Breakout-v0')
	num_actions = env.action_space.n
	# results dir setting
	timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
	exp_name = name or (tag + '-' + timestamp)
	results_dir = os.path.join(f"/data/hanjl/my_results/{env.spec.id}", exp_name)
	summary_dir = os.path.join(results_dir, "tf1_summary")
	if not os.path.exists(summary_dir):
		os.makedirs(summary_dir)
	summary_writer = tf.summary.FileWriter(summary_dir)
	# For episode information
	episode_lengths=np.zeros(num_episodes)
	episode_rewards=np.zeros(num_episodes)
	# Config used for recording all parameters
	config = {}
	config['exp_name'] = exp_name
	config['time'] = timestamp
	config['env'] = env.spec.id
	config['results_dir'] = results_dir
	# Tf.Session()
	gpu_options = tf.GPUOptions(allow_growth=True)	
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		# Instantiate Agent and initialize all variables
		agent = Agent(sess,
					  exp_dir=results_dir,
					  num_actions=num_actions,
					  replay_capacity=1000000,
					  replay_min_size=50000,
					  update_period=4,
					  target_update_period=10000,
					  epsilon_end=0.01,
					  epsilon_decay_period=1000000,
					  gamma=0.99,
					  batch_size=32,
					  # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/RMSPropOptimizer
					  # #migrate-to-tf2_2
					  optimizer=tf.train.RMSPropOptimizer(learning_rate=0.00025, \
						  								  decay=0.95, \
														  epsilon=1e-6, \
														  centered=False),
					  summary_writer=summary_writer,
                      summary_writing_frequency=500)
		sess.run(tf.global_variables_initializer())
		# Save config.json
		config['agent_config'] = agent.config
		config_json = json.dumps(config, sort_keys=False, indent=4, separators=(',', ': '))
		with open(os.path.join(results_dir, "config.json"), 'w') as out:
			out.write(config_json)
		# Start iteration of episodes
		for i in range(num_episodes):
			observation = env.reset()
			agent.reset_history(observation)
			# One episode
			while True:
				print(f"\r@episode: {i}/{num_episodes}, length: {episode_lengths[i]}", end='')
				action = agent.select_action()
				observation, reward, done, _ = env.step(action)
				agent.step(action, observation, reward, done)
				episode_rewards[i] += reward
				episode_lengths[i] += 1
				if done:
					break
			# Summary episode infomation
			episode_summary = tf.Summary(value=[
				tf.Summary.Value(simple_value=episode_rewards[i], tag="episode_info/reward"),
				tf.Summary.Value(simple_value=episode_lengths[i], tag="episode_info/length")
			])
			summary_writer.add_summary(episode_summary, i)
			# summary_writer.flush()
			print(f"\nreward: {episode_rewards[i]}")

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--tag', type=str, default='test', help='Used as exp_name')
	parser.add_argument('--name', type=str, default=None, help='Used as exp_name')
	parser.add_argument('--num_ep', type=int, default=10000, help='Number of episodes')
	args, unknown_args = parser.parse_known_args()
	main(args)