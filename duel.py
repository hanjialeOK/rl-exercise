import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

from lib import plotting
from lib.dqn_utils import *
from collections import deque, namedtuple

# make enviroment
env = gym.envs.make("Breakout-v0")

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]

class Estimator():
	"""
	Q-Value Estimator neural network.

	This network is used for both the Q-Network and the Target Network.
	"""
	def __init__(self, scope="estimator", summaries_dir=None):
		self.scope = scope
		# Writes Tensorboard summaries to disk
		self.summary_writer = None
		with tf.variable_scope(scope):
			# Build the graph
			self._build_model()
			if summaries_dir:
				summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
				if not os.path.exists(summary_dir):
					os.makedirs(summary_dir)
				self.summary_writer = tf.summary.FileWriter(summary_dir)

	def _build_model( self ):
		"""
			build computation graph
		"""

		# Placeholders for our input
		# Our input are 4 RGB frames of shape 84, 84 each
		self.X_pl = tf.placeholder(shape=[32, 4, 84, 84], dtype=tf.uint8, name="X")
		# The TD target value
		self.y_pl = tf.placeholder(shape=[32], dtype=tf.float32, name="y")
		# Integer id of which action was selected
		self.actions_pl = tf.placeholder(shape=[32], dtype=tf.int32, name="actions")

		X = tf.to_float(self.X_pl) / 255.0
		batch_size = tf.shape(self.X_pl)[0]

		# Three convolutional layers
		conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu, data_format='NCHW')
		conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu, data_format='NCHW')
		conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu, data_format='NCHW')

		# Fully connected layers
		flattened = tf.contrib.layers.flatten(conv3)
		fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)
		fc2 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu)

		# value
		self.value = tf.contrib.layers.fully_connected(fc1, 1, activation_fn=None)

		# advantage
		self.advantage = tf.contrib.layers.fully_connected(fc2, len(VALID_ACTIONS), activation_fn=None)

		# predictions
		self.predictions = self.value + (self.advantage - \
								tf.reduce_mean(self.advantage, axis=1, keepdims=True))

		# Get the predictions for the chosen actions only
		gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
		self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

		# Square difference loss
		# self.losses = tf.squared_difference( self.y_pl, self.action_predictions )
		# Huber loss
		self.losses = tf.losses.huber_loss(self.y_pl, self.action_predictions, reduction=tf.losses.Reduction.NONE)

		self.loss = tf.reduce_mean( self.losses )

		# Optimizer Parameters from original paper
		self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, momentum=0, \
																		epsilon=1e-6, centered=False)
		self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

		# Summaries for Tensorboard
		self.summaries = tf.summary.merge([
			tf.summary.scalar("loss", self.loss),
			# tf.summary.histogram("loss_hist", self.losses),
			# tf.summary.histogram("q_values_hist", self.predictions),
			tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
			tf.summary.scalar("avg_q_value", tf.reduce_mean(self.predictions))
		])


	def predict(self, sess, s):
		"""
		Predicts action values.

		Args:
			sess: Tensorflow session
			s: State input of shape [batch_size, 4, 160, 160, 3]

		Returns:
			Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
			action values.
		"""
		return sess.run(self.predictions, { self.X_pl: s })

	def update(self, sess, s, a, y):
		"""
		Updates the estimator towards the given targets.

		Args:
		  sess: Tensorflow session object
		  s: State input of shape [batch_size, 4, 160, 160, 3]
		  a: Chosen actions of shape [batch_size]
		  y: Targets of shape [batch_size]

		Returns:
		  The calculated loss on the batch.
		"""
		feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
		summaries, global_step, _, loss = sess.run(
			[self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss], feed_dict)
		if self.summary_writer:
			self.summary_writer.add_summary(summaries, global_step)
		return loss

def deep_q_learning(sess,
					env,
					q_estimator,
					target_estimator,
					state_processor,
					num_episodes,
					experiment_dir,
					replay_memory_size=500000,
					replay_memory_init_size=50000,
					update_target_estimator_every=10000,
					discount_factor=0.99,
					epsilon_start=1.0,
					epsilon_end=0.1,
					epsilon_decay_steps=500000,
					batch_size=32,
					record_video_every=50):
	"""
	Q-Learning algorithm for off-policy TD control using Function Approximation.
	Finds the optimal greedy policy while following an epsilon-greedy policy.

	Args:
		sess: Tensorflow Session object
		env: OpenAI environment
		q_estimator: Estimator object used for the q values
		target_estimator: Estimator object used for the targets
		state_processor: A StateProcessor object
		num_episodes: Number of episodes to run for
		experiment_dir: Directory to save Tensorflow summaries in
		replay_memory_size: Size of the replay memory
		replay_memory_init_size: Number of random experiences to sampel when initializing 
			the reply memory.
		update_target_estimator_every: Copy parameters from the Q estimator to the 
			target estimator every N steps
		discount_factor: Gamma discount factor
		epsilon_start: Chance to sample a random action when taking an action.
			Epsilon is decayed over time and this is the start value
		epsilon_end: The final minimum value of epsilon after decaying is done
		epsilon_decay_steps: Number of steps to decay epsilon over
		batch_size: Size of batches to sample from the replay memory
		record_video_every: Record a video every N episodes

	Returns:
		An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
	"""

	Transition = namedtuple("Transition", ["action", "next_state", "reward", "done"])

	# The replay memory
	replay_memory = []

	# Keeps track of useful statistics
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))

	# Create directories for checkpoints and summaries
	checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
	checkpoint_path = os.path.join(checkpoint_dir, "model")
	monitor_path = os.path.join(experiment_dir, "monitor")
	with open(os.path.join(experiment_dir, "progress.txt"), 'w') as doc:
		doc.write("Epoch\tAverageEpRet\n")

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	if not os.path.exists(monitor_path):
		os.makedirs(monitor_path)

	saver = tf.train.Saver()
	# Load a previous checkpoint if we find one
	latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
	if latest_checkpoint:
		print("Loading model checkpoint {}...\n".format(latest_checkpoint))
		saver.restore(sess, latest_checkpoint)
		raise

	# Get the current time step
	total_t = sess.run(tf.contrib.framework.get_global_step())

	# The epsilon decay schedule
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	# The policy we're following
	policy = make_epsilon_greedy_policy(
		q_estimator,
		len(VALID_ACTIONS))

	# Populate the replay memory with initial experience
	print("Populating replay memory...")
	############################################################
	# YOUR CODE 1 : Populate replay memory!
	# Hints : use function "populate_replay_buffer"
	# about 1 line code
	replay_memory = populate_replay_buffer(sess, env, state_processor, replay_memory_init_size, \
										   VALID_ACTIONS, Transition, policy)

	# Record videos
	env= Monitor(env,
				directory=monitor_path,
				resume=True,
				video_callable=lambda count: count % record_video_every == 0)
	
	copy_model_parameters(sess, q_estimator, target_estimator)

	# history
	history = History()

	for i_episode in range(num_episodes):
		# Save the current checkpoint
		saver.save(tf.get_default_session(), checkpoint_path)

		# Reset the environment
		state = env.reset()
		state_proc = state_process( sess, state_processor, state )
		for _ in range(4):
			history.add(state_proc)
		loss = None

		# One step in the environment
		for t in itertools.count():
			# Epsilon for this time step
			epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

			# Add epsilon to Tensorboard
			episode_summary = tf.Summary()
			episode_summary.value.add(simple_value=epsilon, tag="epsilon")
			q_estimator.summary_writer.add_summary(episode_summary, total_t)

			# ###########################################################
			# # YOUR CODE 2: Target network update
			# # Hints : use function  "copy_model_parameters"
			# if total_t % update_target_estimator_every == 0:
			# 	copy_model_parameters(sess, q_estimator, target_estimator)

			# Print out which step we're on, useful for debugging.
			print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
					t, total_t, i_episode + 1, num_episodes, loss), end="")
			sys.stdout.flush()

			##############################################
			# YOUR CODE 3: Take a step in the environment
			# Hints 1 :  be careful to use function 'state_process' to deal the RPG state
			# Hints 2 :  you can see function "populate_replay_buffer()" 
			#				for detail about how to TAKE A STEP 
			# about 2 or 3 line codes
			action = np.random.choice( len(VALID_ACTIONS), p=policy( sess, history.get(), epsilon ) )
			next_state, reward, done, _ = env.step( action )

			# If our replay memory is full, pop the first element
			if len(replay_memory) == replay_memory_size:
				replay_memory.pop(0)

			#############################
			# YOUR CODE 4: Save transition to replay memory
			#  Hints : you can see function 'populate_replay_buffer' for detail
			# about 1 or 2 line codes
			next_state_proc = state_process( sess, state_processor, next_state )
			history.add(next_state_proc)
			experience = Transition(action, next_state_proc, reward, done )
			replay_memory.append( experience )

			# Update statistics
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			if total_t % 4 == 0:
				#########################################################
				# YOUR CODE 5: Sample a minibatch from the replay memory, 
				# hints: can use function "random.sample( replay_memory, batch_size )" to get minibatch
				# about 1-2 lines codes
				indexes = []
				states = np.zeros(shape=(batch_size, 4, 84, 84), dtype=np.uint8)
				next_states = np.zeros(shape=(batch_size, 4, 84, 84), dtype=np.uint8)
				while len(indexes) < batch_size:
					while True:
						index = random.randint(4, len(replay_memory) - 1)
						dones_before_index = \
							np.stack([trans[3] for trans in replay_memory[(index-4):index]], axis=0).astype(np.uint32)
						if dones_before_index.any():
							continue
						break
					states[len(indexes)] = \
						np.stack([trans[1] for trans in replay_memory[(index-4):index]], axis=0).astype(np.uint8)
					next_states[len(indexes)] = \
						np.stack([trans[1] for trans in replay_memory[(index-3):(index+1)]], axis=0).astype(np.uint8)
					indexes.append(index)
				actions = np.stack([replay_memory[i][0] for i in indexes], axis=0).astype(np.int32) # (32, )
				rewards = np.stack([replay_memory[i][2] for i in indexes], axis=0).astype(np.int32) # (32, )
				dones = np.stack([replay_memory[i][3] for i in indexes], axis=0).astype(np.int32) # (32, )

				###########################################################
				# YOUR CODE 6: use minibatch sample to calculate q values and targets
				# Hints 1 : use function 'q_estimator.predict' to get q values
				# Hints 2 : use function 'target_estimator.predict' to get targets values
				#				remember 'targe = reward + gamma * max q( s, a' )'
				# about 2 line codes
				t_values = target_estimator.predict(sess, next_states) # (32, 4)
				pred = rewards + (1 - dones) * discount_factor * np.max(t_values, axis=1, keepdims=False) # (32, )

				################################################
				# YOUR CODE 7: Perform gradient descent update
				# hints : use function 'q_estimator.update'
				# about 1 line code
				loss = q_estimator.update(sess=sess, s=states, a=actions, y=pred)

			###########################################################
			# YOUR CODE 2: Target network update
			# Hints : use function  "copy_model_parameters"
			if total_t % update_target_estimator_every == update_target_estimator_every - 1:
				copy_model_parameters(sess, q_estimator, target_estimator)

			total_t += 1

			if done:
				break
		
		with open(os.path.join(experiment_dir, "progress.txt"), 'a') as doc:
			doc.write(f"{i_episode}\t{stats.episode_rewards[i_episode]}\n")

		# Add summaries to tensorboard
		episode_summary = tf.Summary()
		episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
		episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
		q_estimator.summary_writer.add_summary(episode_summary, total_t)
		q_estimator.summary_writer.flush()

		yield total_t, plotting.EpisodeStats(
			episode_lengths=stats.episode_lengths[:i_episode+1],
			episode_rewards=stats.episode_rewards[:i_episode+1])

	env.close()
	return stats

tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id + "-" + \
									time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

# Run it!
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for t, stats in deep_q_learning(sess,
									env,
									q_estimator=q_estimator,
									target_estimator=target_estimator,
									state_processor=state_processor,
									experiment_dir=experiment_dir,
									num_episodes=10000,
									replay_memory_size=1000000,
									replay_memory_init_size=50000,
									update_target_estimator_every=10000,
									epsilon_start=1.0,
									epsilon_end=0.01,
									epsilon_decay_steps=1000000,
									discount_factor=0.99,
									batch_size=32):
		print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
	# plot_episode_stats(stats)
