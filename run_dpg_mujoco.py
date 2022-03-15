import numpy as np
import tensorflow as tf
import gym
import time
import os
import argparse

import dpg.agents.ddpg as DDPG
import dpg.agents.td3 as TD3
import dpg.agents.sac as SAC

from termcolor import cprint, colored
from common.serialization_utils import convert_json, save_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(env_eval, agent, num_ep=10):
    ret_sum = 0.
    len_sum = 0

    for i in range(1, num_ep + 1):
        obs = env_eval.reset()
        ep_ret = 0.
        ep_len = 0
        while True:
            ac = agent.select_action(obs)
            obs, reward, done, _ = env_eval.step(ac, noise=False)
            ep_ret += reward
            ep_len += 1
            if done:
                cprint(f'\rEvaluate: {i}/{num_ep}',
                       color='cyan', attrs=['bold'], end='')
                ret_sum += ep_ret
                len_sum += ep_len
                break
    avg_eval_ret = ret_sum / num_ep
    avg_eval_len = len_sum / num_ep
    cprint(f'\navg_eval_len: {ep_len}, avg_eval_ret: {ep_ret:.1f}',
           color='cyan', attrs=['bold'])

    return avg_eval_ret, avg_eval_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default=None, help='Dir name')
    parser.add_argument('--data_dir', type=str, default='/data/hanjl',
                        help='Data disk dir')
    parser.add_argument('--env_name', '--env', type=str,
                        default='HalfCheetah-v2')
    parser.add_argument('--exp_name', type=str, default='DDPG',
                        choices=['DPG', 'DDPG', 'TD3', 'SAC'],
                        help='Experiment name',)
    parser.add_argument('--allow_eval', action='store_true',
                        help='Whether to eval agent')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise

    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    dir_name = args.dir_name or (args.exp_name + '-' + timestamp)
    exp_name = args.exp_name
    env_name = args.env_name
    base_dir = os.path.join(args.data_dir, f"my_results/{env_name}/{dir_name}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Dump config.
    # locals() can only be put at main(), instead of __main__.
    config = convert_json(locals())
    save_json(config, base_dir)

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

    # Random seed
    seed = int(time.time()) % 1000
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    # Environment
    env = gym.make(env_name)
    env.seed(seed)
    env_eval = gym.make(env_name)
    env_eval.seed(seed + 1)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    max_action = float(env.action_space.high[0])

    # Tensorboard
    summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    # Experience buffer
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False))

    # Tensorflow message must be put after sess.
    # DEBUG(0), INFO(1), WARNING(2), ERROR(3)
    # Combine os.environ['TF_CPP_MIN_LOG_LEVEL'] and tf.get_logger()
    tf.get_logger().setLevel('ERROR')
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    cprint(f'Tensorflow version: {tf.__version__}, '
           f'GPU Available: {tf.test.is_gpu_available()}, '
           f'GPU count: {len(gpu_list)}\n'
           f'{gpu_list}\n',
           color='cyan', attrs=['bold'])

    if exp_name == 'DPG':
        raise NotImplementedError
    elif exp_name == 'DDPG':
        agent = DDPG.DDPGAgent(sess, obs_dim, act_dim,
                               max_action, noise_scale=0.1)
    elif exp_name == 'TD3':
        agent = TD3.TD3Agent(sess, obs_dim, act_dim,
                             max_action, noise_scale=0.1)
    elif exp_name == 'SAC':
        agent = SAC.SACAgent(sess, obs_dim, act_dim, max_action)
    else:
        raise ValueError(f'Unknown agent: {exp_name}')

    sess.run(tf.compat.v1.global_variables_initializer())
    agent.target_params_init()

    # Params
    total_steps = int(1e6)
    start_steps = int(25e3)
    eval_freq = int(5e3)

    cprint(f'Running experiment: {exp_name}\n', color='cyan', attrs=['bold'])

    # Start
    start_time = time.time()
    obs = env.reset()
    ep_ret, ep_len = 0, 0
    ep_count = 0

    for t in range(1, total_steps + 1):
        if t > start_steps:
            ac = agent.select_action(obs, noise=True)
        else:
            ac = env.action_space.sample()

        next_obs, reward, done, _ = env.step(ac)
        ep_ret += reward
        ep_len += 1

        terminal = False if ep_len == env._max_episode_steps else done

        agent.store_transition(obs, next_obs, ac, reward, terminal)

        obs = next_obs

        if done:
            # Episode summary
            ep_count += 1
            episode_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag="episode_info/reward", simple_value=ep_ret),
                tf.compat.v1.Summary.Value(
                    tag="episode_info/length", simple_value=ep_len)
            ])
            summary_writer.add_summary(episode_summary, ep_count)
            ep_ret_text = colored(f'{ep_ret:.1f}',
                                  color='green', attrs=['bold'])
            print(f'Training {exp_name}: {t/total_steps:.1%}, '
                  f'ep_len: {ep_len}, '
                  f'ep_ret: {ep_ret_text}, ')
            # Episode restart
            obs = env.reset()
            ep_ret, ep_len = 0, 0

        if t > start_steps:
            agent.update()

        # Evaluate
        if t % eval_freq == 0 and args.allow_eval:
            avg_ret, evg_len = evaluate(
                env_eval, agent, num_ep=10)
            # Summary
            eval_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag='eval/avg_len', simple_value=evg_len),
                tf.compat.v1.Summary.Value(
                    tag='eval/avg_reward', simple_value=avg_ret)
            ])
            summary_writer.add_summary(eval_summary, t)
            # Save the best weights
            if avg_ret >= max_ep_ret and args.save_model:
                print(f'Saving weights into {checkpoint_dir}')
                agent.bundle(checkpoint_dir, t // eval_freq)
                max_ep_ret = avg_ret
            # Log data
            with open(progress_txt, 'a') as f:
                f.write(f"{t}\t{avg_ret}\n")

    time_delta = int(time.time() - start_time)
    m, s = divmod(time_delta, 60)
    h, m = divmod(m, 60)
    print(f'Time taken: {h:d}:{m:02d}:{s:02d}')
    print(f"Results saved into {base_dir}")
    summary_writer.flush()
    env.close()
    env_eval.close()


if __name__ == '__main__':
    main()
