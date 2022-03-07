import numpy as np
import os
import tensorflow as tf
import time
from argparse import ArgumentParser
from collections import namedtuple

from utils.serialization_utils import convert_json, save_config
from deepq.agents.dqn import *
from deepq.agents.rainbow import *
from deepq.env.atari_lib import create_atari_environment


def create_agent(sess, num_actions, exp_name=None, summary_writer=None):
    assert exp_name is not None
    if exp_name == 'dqn':
        return DQNAgent(
            sess=sess, num_actions=num_actions, summary_writer=summary_writer)
    if exp_name == 'clipdqn':
        return ClippedDQN(
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
    elif exp_name == 'c51':
        return C51Agent(
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
        self.config = convert_json(locals())
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
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                      log_device_placement=True))
        # env
        self._env = create_env_fn(env_name)
        num_actions = self._env.action_space.n
        # agent
        self._agent = create_agent_fn(sess=self._sess, num_actions=num_actions,
                                      exp_name=exp_name, summary_writer=self._summary_writer)
        self._summary_writer.add_graph(graph=tf.get_default_graph())
        self._sess.run(tf.global_variables_initializer())

        self.config['env_config'] = self._env.config
        self.config['agent_config'] = self._agent.config

        self._iteration = 0
        self._total_episodes = 0
        self._max_episode_reward = 0
        self.PhaseDataType = namedtuple('PhaseDataType',
                                        ['num_episodes', 'average_reward', 'average_steps_per_second'])

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
            print(
                f"\rlength: {episode_length}, reward: {episode_reward}", end='')

            action = self._agent.select_action()
            observation, reward, terminal, _ = self._env.step(action)

            episode_reward += reward
            episode_length += 1

            reward_clip = np.clip(reward, -1, 1)

            if self._env.was_real_done or (episode_length >= self._max_steps_per_episode):
                # we lose all lifes
                self._agent.step(action, observation, reward_clip, True)
                break
            elif terminal:
                # If we lose a life but the episode is not over
                # Terminal on life loss = True
                self._agent.step(action, observation, reward_clip, True)
                observation = self._env.reset()
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
            print(f"\n@iter: {self._iteration}/{self._num_iterations}, train: {not eval_mode}, "
                  f"step: {step_count}/{min_steps}, total_ep: {self._total_episodes}")
            episode_length, episode_reward = self._run_one_episode()
            step_count += episode_length
            sum_rewards += episode_reward
            num_episodes += 1
            # episode_info
            if not eval_mode:
                episode_summary = tf.Summary(value=[
                    tf.Summary.Value(simple_value=episode_reward,
                                     tag="episode_info/reward"),
                    tf.Summary.Value(simple_value=episode_length,
                                     tag="episode_info/length")
                ])
                self._summary_writer.add_summary(
                    episode_summary, self._total_episodes)
                self._total_episodes += 1
        time_delta = time.time() - start_time
        average_steps_per_second = step_count / time_delta
        average_reward = sum_rewards / num_episodes
        return self.PhaseDataType(num_episodes, average_reward, average_steps_per_second)

    def _save_tensorboard_summaries(self, train_data, eval_data):
        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='train/num_episodes', simple_value=train_data.num_episodes),
            tf.Summary.Value(
                tag='train/average_reward', simple_value=train_data.average_reward),
            tf.Summary.Value(
                tag='train/average_steps_per_second',
                simple_value=train_data.average_steps_per_second),
            tf.Summary.Value(
                tag='eval/num_episodes', simple_value=eval_data.num_episodes),
            tf.Summary.Value(
                tag='eval/average_reward', simple_value=eval_data.average_reward)
        ])
        self._summary_writer.add_summary(summary, self._iteration)

    # Save 4 best models according to eval_data.average_reward.
    def _checkpoint_experiment(self, eval_data):
        if eval_data.average_reward >= self._max_episode_reward:
            print(f'\nSaving weights into {self._checkpoint_dir}')
            self._agent.bundle(self._checkpoint_dir, self._iteration)
            self._max_episode_reward = eval_data.average_reward

    def _log_experiment(self, eval_data):
        with open(self._progress_txt, 'a') as f:
            f.write(f"{self._iteration}\t{eval_data.average_reward}\n")

    def run_experiment(self):
        for self._iteration in range(self._num_iterations):
            train_data = self._run_one_phase(min_steps=self._min_train_steps)
            eval_data = self._run_one_phase(
                min_steps=self._evaluation_steps, eval_mode=True)
            self._save_tensorboard_summaries(train_data, eval_data)
            self._checkpoint_experiment(eval_data)
            self._log_experiment(eval_data)
        print(f"\nResults have been saved into {self._base_dir}")
        self._summary_writer.flush()
        self._env.close()

    def test_restore(self, checkpoint_dir, iteration=None):
        print('Loading weights from {}'
              .format(os.path.join(checkpoint_dir, f'tf_ckpt-{iteration}')))
        self._agent.unbundle(checkpoint_dir, iteration)
        for _ in range(10):
            self._run_one_phase(
                min_steps=self._evaluation_steps, eval_mode=True)


class StepAwareRunner():
    def __init__(self,
                 base_dir,
                 exp_name,
                 env_name,
                 create_agent_fn=create_agent,
                 create_env_fn=create_atari_environment,
                 total_steps=int(12e6),
                 num_iterations=200,
                 max_steps_per_episode=20000,
                 clip_rewards=True):
        if not os.path.join(base_dir):
            raise
        self.config = convert_json(locals())
        self._base_dir = base_dir
        self._total_steps = total_steps
        self._num_iterations = num_iterations
        self._min_train_steps = int(total_steps / num_iterations)
        # The larger the number of evaluation steps, The smoother the curve is.
        self._evaluation_steps = int(self._min_train_steps / 2)
        self._max_steps_per_episode = max_steps_per_episode
        self._clip_rewards = clip_rewards
        # create dirs
        self._create_directories()
        # summary_writer
        self._summary_writer = tf.summary.FileWriter(self._summary_dir)
        # sess
        gpu_options = tf.GPUOptions(allow_growth=True)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                      log_device_placement=True))
        # env
        self._env = create_env_fn(env_name)
        num_actions = self._env.action_space.n
        # agent
        self._agent = create_agent_fn(sess=self._sess, num_actions=num_actions,
                                      exp_name=exp_name, summary_writer=self._summary_writer)
        self._summary_writer.add_graph(graph=tf.get_default_graph())
        self._sess.run(tf.global_variables_initializer())

        # self.config['env_config'] = self._env.config
        self.config['agent_config'] = self._agent.config

        self._iteration = 0
        self._total_episodes = 0
        self._max_episode_reward = 0

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
            print(
                f"\rlength: {episode_length}, reward: {episode_reward}", end='')

            action = self._agent.select_action()
            observation, reward, terminal, _ = self._env.step(action)

            episode_reward += reward
            episode_length += 1

            # reward_clip = np.clip(reward, -1, 1)

            if terminal or (episode_length >= self._max_steps_per_episode):
                break
            elif self._env.was_life_loss:
                # If we lose a life but the episode is not over
                observation = self._env.reset()
                self._agent.begin_episode(observation)
            else:
                self._agent.step(action, observation, reward, False)
        return episode_length, episode_reward

    def _evaluate(self, min_steps):
        step_count = 0
        num_episodes = 0
        sum_rewards = 0.

        self._agent.eval_mode = True

        while step_count < min_steps:
            print(f"\n@iter: {self._iteration}/{self._num_iterations}, train: {False}, "
                  f"step: {step_count}/{min_steps}, total_ep: {self._total_episodes}")
            episode_length, episode_reward = self._run_one_episode()
            step_count += episode_length
            sum_rewards += episode_reward
            num_episodes += 1

        self._agent.eval_mode = False

        average_reward = sum_rewards / num_episodes
        return num_episodes, average_reward

    def run_experiment(self):
        episode_length = 0
        episode_reward = 0.
        train_sum_rewards = 0.
        train_num_episodes = 0
        start_time = time.time()

        observation = self._env.reset()
        self._agent.begin_episode(observation)
        for step in range(1, self._total_steps + 1):
            print(
                f"\rlength: {episode_length}, reward: {episode_reward}", end='')

            action = self._agent.select_action()
            observation, reward, terminal, _ = self._env.step(action)

            episode_reward += reward
            episode_length += 1

            reward_clip = np.clip(reward, -1, 1)

            if terminal or (episode_length >= self._max_steps_per_episode):
                # Lose all lives
                self._agent.step(action, observation, reward_clip, True)
                # Summary
                episode_summary = tf.Summary(value=[
                    tf.Summary.Value(simple_value=episode_reward,
                                     tag="episode_info/reward"),
                    tf.Summary.Value(simple_value=episode_length,
                                     tag="episode_info/length")
                ])
                self._summary_writer.add_summary(
                    episode_summary, self._total_episodes)
                self._total_episodes += 1
                # Train data statistics
                train_num_episodes += 1
                train_sum_rewards += episode_reward
                print(f"\n@iter: {self._iteration}/{self._num_iterations}, train: {True}, "
                      f"step: {step % self._min_train_steps}/{self._min_train_steps}, "
                      f"total_ep: {self._total_episodes}")
                # Episode restart
                episode_length = 0
                episode_reward = 0.
                observation = self._env.reset()
                self._agent.begin_episode(observation)
            elif self._env.was_life_loss:
                # If we lose a life but the episode is not over
                self._agent.step(action, observation, reward_clip, True)
                observation = self._env.reset()
                self._agent.begin_episode(observation)
            else:
                self._agent.step(action, observation, reward_clip, False)

            if step % self._min_train_steps == 0:
                # Train data statistics
                time_delta = time.time() - start_time
                train_average_steps_per_second = self._min_train_steps / time_delta
                train_average_reward = train_sum_rewards / train_num_episodes
                # Evaluate
                eval_num_episodes, eval_average_reward = self._evaluate(
                    self._evaluation_steps)
                # Summary
                summary = tf.Summary(value=[
                    tf.Summary.Value(
                        tag='train/num_episodes', simple_value=train_num_episodes),
                    tf.Summary.Value(
                        tag='train/average_reward', simple_value=train_average_reward),
                    tf.Summary.Value(
                        tag='train/average_steps_per_second',
                        simple_value=train_average_steps_per_second),
                    tf.Summary.Value(
                        tag='eval/num_episodes', simple_value=eval_num_episodes),
                    tf.Summary.Value(
                        tag='eval/average_reward', simple_value=eval_average_reward)
                ])
                self._summary_writer.add_summary(summary, step)
                # Save the best weights
                if eval_average_reward >= self._max_episode_reward:
                    print(f'\nSaving weights into {self._checkpoint_dir}')
                    self._agent.bundle(self._checkpoint_dir, self._iteration)
                    self._max_episode_reward = eval_average_reward
                # Log data
                with open(self._progress_txt, 'a') as f:
                    f.write(f"{step}\t{eval_average_reward}\n")
                # Return zero
                train_sum_rewards = 0.
                train_num_episodes = 0
                start_time = time.time()
                self._iteration += 1
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
    base_dir = os.path.join(args.disk_dir, f"stepaware/{env_name}/{dir_name}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    config = convert_json(locals())
    # Runner
    runner = StepAwareRunner(
        base_dir=base_dir, exp_name=exp_name, env_name=env_name)
    config['runner_config'] = runner.config
    # Save config_json
    save_config(config, base_dir)
    # Run
    runner.run_experiment()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir_name', type=str, default=None, help='Dir name')
    parser.add_argument('--exp_name', type=str, default='dqn', help='Experiment name',
                        choices=['dqn', 'clipdqn', 'ddqn', 'prior', 'duel', 'c51',
                                 'ddqn+prior', 'ddqn+duel'])
    parser.add_argument('--env_name', type=str,
                        default='Breakout', help='Env name')
    parser.add_argument('--sticky', action='store_true', help='Sticky actions')
    parser.add_argument('--disk_dir', type=str,
                        default='/data/hanjl', help='Data disk dir')
    args, unknown_args = parser.parse_known_args()
    main(args)
