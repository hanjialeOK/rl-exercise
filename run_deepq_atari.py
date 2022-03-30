import numpy as np
import os
import tensorflow as tf
import time
from argparse import ArgumentParser
from termcolor import cprint, colored

import deepq.agents.dqn as DQN
import deepq.agents.ddqn as DDQN
import deepq.agents.per as PER
import deepq.agents.dueling as Duel
import deepq.agents.rainbow as C51

from common.serialization_utils import convert_json, save_json
from deepq.env.atari_lib import create_atari_environment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(agent, env, n_eval_episodes=10, max_steps=20000):
    sum_rewards = 0.
    sum_len = 0

    agent.eval_mode = True

    for i in range(1, n_eval_episodes + 1):
        episode_reward = 0.
        episode_length = 0
        obs = env.reset()
        agent.begin_episode(obs)
        while True:

            action = agent.select_action()
            obs, reward, done, _ = env.step(action)

            agent.update_observation(obs)

            episode_reward += reward
            episode_length += 1

            if done or (episode_length >= max_steps):
                break
        sum_len += episode_length
        sum_rewards += episode_reward

    agent.eval_mode = False

    avgret = sum_rewards / n_eval_episodes
    avglen = sum_len / n_eval_episodes

    return avgret, avglen


def main():
    parser = ArgumentParser()
    parser.add_argument('--dir_name', type=str,
                        default='default', help='Dir name')
    parser.add_argument('--alg', type=str, default='dqn', help='Algorithm name',
                        choices=['dqn', 'clipdqn', 'ddqn', 'per', 'duel', 'c51'])
    parser.add_argument('--env', type=str,
                        default='Breakout', help='Env name')
    parser.add_argument('--sticky', action='store_true', help='Sticky actions')
    parser.add_argument('--data_dir', type=str,
                        default='/data/hanjl', help='Data disk dir')
    parser.add_argument('--allow_eval', action='store_true',
                        help='Whether to eval agent')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model')
    parser.add_argument('--total_steps', type=int, default=12e6,
                        help='Total steps trained')
    args, unknown_args = parser.parse_known_args()

    if not os.path.exists(args.data_dir):
        raise

    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    base_name = args.alg + '-' + timestamp
    dir_name = args.dir_name
    exp_name = args.alg
    env_name = f"{args.env}NoFrameskip-{'v0' if args.sticky else 'v4'}"
    base_dir = os.path.join(
        args.data_dir, f"my_results/{env_name}/{dir_name}/{base_name}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    config = convert_json(locals())
    save_json(config, base_dir)

    summary_dir = os.path.join(base_dir, "tf1_summary")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    train_txt = os.path.join(base_dir, 'train.txt')
    eval_txt = os.path.join(base_dir, 'eval.txt')
    with open(train_txt, 'w') as f1, open(eval_txt, 'w') as f2:
        f1.write('Step\tAvgReturn\n')
        f2.write('Step\tAvgReturn\n')

    # Random seed
    seed = int(time.time()) % 1000
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    # Environment
    env = create_atari_environment(env_name)
    env_eval = create_atari_environment(env_name)
    num_actions = env.action_space.n

    # Summary writer
    summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    # Session
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False))

    # Tensorflow message must be put after sess.
    tf.get_logger().setLevel('ERROR')
    gpu_list = tf.compat.v1.config.experimental.list_physical_devices('GPU')
    cprint(f'Tensorflow version: {tf.__version__}, '
           f'GPU available: {tf.test.is_gpu_available()}, '
           f'GPU count: {len(gpu_list)}\n'
           f'{gpu_list}\n',
           color='cyan', attrs=['bold'])

    # Agent
    if exp_name == 'dqn':
        agent = DQN.DQNAgent(
            sess=sess, num_actions=num_actions, summary_writer=summary_writer)
    elif exp_name == 'clipdqn':
        agent = DQN.ClippedDQN(
            sess=sess, num_actions=num_actions, summary_writer=summary_writer)
    elif exp_name == 'ddqn':
        agent = DDQN.DDQNAgent(
            sess=sess, num_actions=num_actions, summary_writer=summary_writer)
    elif exp_name == 'per':
        agent = PER.PERAgent(
            sess=sess, num_actions=num_actions, summary_writer=summary_writer)
    elif exp_name == 'duel':
        agent = Duel.DuelingAgent(
            sess=sess, num_actions=num_actions, summary_writer=summary_writer)
    elif exp_name == 'c51':
        agent = C51.C51Agent(
            sess=sess, num_actions=num_actions, summary_writer=summary_writer)
    else:
        raise ValueError(f'Unknown agent: {exp_name}')

    summary_writer.add_graph(graph=tf.compat.v1.get_default_graph())
    sess.run(tf.compat.v1.global_variables_initializer())

    total_steps = int(args.total_steps)
    epochs = 200
    min_train_steps = int(total_steps / epochs)
    max_steps_per_episode = 20000
    total_episodes = 0
    max_ep_ret = 0
    start_time = time.time()

    obs = env.reset()
    agent.begin_episode(obs)

    for epoch in range(1, epochs + 1):
        ep_len = 0
        ep_ret = 0.
        train_sum_rewards = 0.
        train_sum_lengths = 0
        train_num_episodes = 0
        train_start_time = time.time()
        for i in range(1, min_train_steps + 1):

            action = agent.select_action()
            obs, reward, done, _ = env.step(action)
            reward_clip = np.clip(reward, -1, 1)

            agent.store_transition(action, obs, reward_clip, done)
            agent.update_observation(obs)
            agent.step()

            ep_ret += reward
            ep_len += 1

            if done or (ep_len >= max_steps_per_episode):
                # Lose all lives
                # agent.step()
                # Train data statistics
                train_num_episodes += 1
                train_sum_lengths += ep_len
                train_sum_rewards += ep_ret
                ep_ret_text = colored(f'{ep_ret:.1f}',
                                      color='green', attrs=['bold'])
                print(f"@Epoch: {epoch}/{epochs}: {epoch/epochs:.1%}, "
                      f"Len: {ep_len}, Ret: {ep_ret_text}, "
                      f"Step: {i}/{min_train_steps}: {i/min_train_steps:.1%}, "
                      f"Algo: {exp_name}")
                # Episode restart
                ep_len = 0
                ep_ret = 0.
                obs = env.reset()
                agent.begin_episode(obs)
            # elif env.was_life_loss:
            #     # If we lose a life but the episode is not over
            #     agent.step()
            #     obs = env.reset()
            #     agent.begin_episode(obs)
            #     raise
            # else:
            #     agent.step()

        # Step trained
        step = epoch * min_train_steps

        # Train data statistics
        train_time_delta = time.time() - train_start_time
        fps = min_train_steps / train_time_delta
        avglen = train_sum_lengths / train_num_episodes
        avgret = train_sum_rewards / train_num_episodes
        # Summary
        train_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='train/num_episodes', simple_value=train_num_episodes),
            tf.Summary.Value(
                tag='train/avglen', simple_value=avglen),
            tf.Summary.Value(
                tag='train/avgret', simple_value=avgret),
            tf.Summary.Value(
                tag='train/fps', simple_value=fps)
        ])
        summary_writer.add_summary(train_summary, step)
        # Log data
        with open(train_txt, 'a') as f:
            f.write(f"{step}\t{avgret}\n")
        m, s = divmod(int(train_time_delta), 60)
        h, m = divmod(m, 60)
        cprint(f"@Epoch: {epoch}/{epochs}: complete, "
               f"AvgLen: {avglen:.1f}, AvgRet: {avgret:.1f}, "
               f'Time: {h:d}:{m:02d}:{s:02d}',
               color='cyan', attrs=['bold'])

        if args.allow_eval:
            # Evaluate
            eval_avgret, eval_avglen = evaluate(agent, env=env_eval)
            # Summary
            eval_summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag='eval/avglen', simple_value=eval_avglen),
                tf.Summary.Value(
                    tag='eval/avgret', simple_value=eval_avgret)
            ])
            summary_writer.add_summary(eval_summary, step)
            # Save the best weights
            if eval_avgret >= max_ep_ret and args.save_model:
                print(f'\nSaving weights into {checkpoint_dir}')
                agent.bundle(checkpoint_dir, epoch)
                max_ep_ret = eval_avgret
            # Log data
            with open(eval_txt, 'a') as f:
                f.write(f"{step}\t{eval_avgret}\n")

    time_delta = int(time.time() - start_time)
    m, s = divmod(time_delta, 60)
    h, m = divmod(m, 60)
    print(f'Time taken: {h:d}:{m:02d}:{s:02d}')
    print(f"\nResults saved into {base_dir}")
    summary_writer.flush()
    env.close()
    env_eval.close()


if __name__ == '__main__':
    main()
