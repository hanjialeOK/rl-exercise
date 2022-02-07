import numpy as np
import os
import tensorflow as tf
import time
import json
from argparse import ArgumentParser
from collections import namedtuple

from lib.utils import json_serializable
from lib.agents.dqn_agent import *
from lib.env.atari import create_atari_environment

def create_agent(sess, num_actions, exp_name=None, summary_writer=None):
	assert exp_name is not None
	if exp_name == 'dqn':
		return DQNAgent(
			sess=sess, num_actions=num_actions, summary_writer=summary_writer)
	elif exp_name == 'ddqn':
		return DDQNAgent(
			sess=sess, num_actions=num_actions, summary_writer=summary_writer)
	elif exp_name == 'prior':
		return PERAgent(
			sess=sess, num_actions=num_actions, summary_writer=summary_writer)
	elif exp_name == 'duel':
		return DuelingAgent(
			sess=sess, num_actions=num_actions, summary_writer=summary_writer)
	else:
		raise ValueError('Unknown agent: {}'.format(exp_name))

class Runner():
	def __init__(self,
				 base_dir,
				 exp_name,
				 env_name,
				 create_agent_fn=create_agent,
				 create_env_fn=create_atari_environment,
				 num_iterations=200,
				 min_train_steps=60000,
				 evaluation_steps=20000,
				 max_steps_per_episode=20000,
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
		# create dirs
		self._create_directories()
		# summary_writer
		self._summary_writer = tf.summary.FileWriter(self._summary_dir)
		# sess
		gpu_options = tf.GPUOptions(allow_growth=True)
		self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		# env
		self._env = create_env_fn(env_name)
		num_actions = self._env.action_space.n
		# agent
		self._agent = create_agent_fn(sess=self._sess, num_actions=num_actions, \
			 						  exp_name=exp_name, summary_writer=self._summary_writer)
		self._summary_writer.add_graph(graph=tf.get_default_graph())
		self._sess.run(tf.global_variables_initializer())

		self.config['env_config'] = self._env.config
		self.config['agent_config'] = self._agent.config

		self.iteration = 0
		self.total_episode = 0
		self.PhaseDataType = namedtuple('PhaseDataType', \
			['num_episodes', 'average_rewrd', 'average_steps_per_second'])

	def _create_directories(self):
		self._summary_dir = os.path.join(self._base_dir, "tf1_summary")
		if not os.path.exists(self._summary_dir):
			os.makedirs(self._summary_dir)
		self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints')
		if not os.path.exists(self._checkpoint_dir):
			os.makedirs(self._checkpoint_dir)
		self._progress_txt = os.path.join(self._base_dir, 'progress.txt')
		with open(self._progress_txt, 'w') as f:
			f.write('Iteration\tValue\n')

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
		return self.PhaseDataType(num_episodes, average_rewrd, average_steps_per_second)

	def _save_tensorboard_summaries(self, train_data, eval_data):
		summary = tf.Summary(value=[
			tf.Summary.Value(
				tag='train/num_episodes', simple_value=train_data.num_episodes),
			tf.Summary.Value(
				tag='train/average_reward', simple_value=train_data.average_rewrd),
			tf.Summary.Value(
				tag='train/average_steps_per_second',
				simple_value=train_data.average_steps_per_second),
			tf.Summary.Value(
				tag='eval/num_episodes', simple_value=eval_data.num_episodes),
			tf.Summary.Value(
				tag='eval/average_reward', simple_value=eval_data.average_rewrd)
		])
		self._summary_writer.add_summary(summary, self.iteration)

	def _checkpoint_experiment(self):
		self._agent.bundle(self._checkpoint_dir, self.iteration)

	def _log_experiment(self, eval_data):
		with open(self._progress_txt, 'a') as f:
			f.write(f"{self.iteration}\t{eval_data.average_rewrd}\n")

	def run_experiment(self):
		for self.iteration in range(self._num_iterations):
			train_data = self._run_one_phase(min_steps=self._min_train_steps)
			eval_data = self._run_one_phase(min_steps=self._evaluation_steps, eval_mode=True)
			self._save_tensorboard_summaries(train_data, eval_data)
			self._checkpoint_experiment()
			self._log_experiment(eval_data)
		print(f"\nResults have been saved into {self._base_dir}")
		self._summary_writer.flush()
		self._env.close()

def main(args):
	if not os.path.exists(args.disk_dir):
		raise
	timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
	dir_name = args.dir_name or (args.exp_name + '-' + timestamp)
	exp_name = args.exp_name
	env_name = f"{args.env_name}NoFrameskip-{'v0' if args.sticky else 'v4'}"
	base_dir = os.path.join(args.disk_dir, f"my_results/{env_name}/{dir_name}")
	if not os.path.exists(base_dir):
		os.makedirs(base_dir)
	config = json_serializable(locals())
	# Runner
	runner = Runner(base_dir=base_dir, exp_name=exp_name, env_name=env_name)
	config['runner_config'] = runner.config
	# Save config_json
	config_json = json.dumps(config, sort_keys=False, indent=4, separators=(',', ': '))
	with open(os.path.join(base_dir, "config.json"), 'w') as out:
		out.write(config_json)
	# Run
	runner.run_experiment()

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--dir_name', type=str, default=None, help='Dir name')
	parser.add_argument('--exp_name', type=str, default='dqn', help='Experiment name', \
						choices=['dqn', 'ddqn', 'prior', 'duel', \
								 'ddqn+prior', 'ddqn+duel'])
	parser.add_argument('--env_name', type=str, default='Breakout', help='Env name')
	parser.add_argument('--sticky', action='store_true', help='Sticky actions')
	parser.add_argument('--disk_dir', type=str, default='/data/hanjl', help='Data disk dir')
	args, unknown_args = parser.parse_known_args()
	main(args)