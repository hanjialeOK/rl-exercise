import numpy as np
import tensorflow as tf
import gym
import time
import os
import argparse
import json

import pg.agents.trpo as TRPO
import pg.agents.ppo as PPO
import pg.agents.ppo2 as PPO2

from termcolor import cprint
from utils.serialization_utils import convert_json, save_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

EPS = 1e-8


def evaluate(env_eval, agent, num_ep=10, max_ep_len=10000):
    ret_sum = 0.
    len_sum = 0

    for i in range(1, num_ep + 1):
        obs = env_eval.reset()
        ep_ret = 0.
        ep_len = 0
        while True:
            ac = agent.select_action(obs)
            obs, reward, done, _ = env_eval.step(ac)
            ep_ret += reward
            ep_len += 1
            if done or ep_len >= max_ep_len:
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
    parser.add_argument('--disk_dir', type=str, default='/data/hanjl',
                        help='Data disk dir')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--exp_name', type=str, default='PPO',
                        choices=['TRPO', 'PPO', 'PPO2'],
                        help='Experiment name',)
    parser.add_argument('--allow_eval', action='store_true',
                        help='Whether to eval agent')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model')
    args = parser.parse_args()

    if not os.path.exists(args.disk_dir):
        raise

    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    dir_name = args.dir_name or (args.exp_name + '-' + timestamp)
    exp_name = args.exp_name
    env_name = args.env_name
    base_dir = os.path.join(args.disk_dir, f"my_results/{env_name}/{dir_name}")
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
    env_eval.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Tensorboard
    summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    # Experience buffer
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,
            allow_soft_placement=False))

    if exp_name == 'TRPO':
        agent = TRPO.TRPOAgent(sess, obs_dim, act_dim, horizon=1000)
    elif exp_name == 'PPO':
        agent = PPO.PPOAgent(sess, obs_dim, act_dim, horizon=1000)
    elif exp_name == 'PPO2':
        agent = PPO2.PPOAgent(sess, obs_dim, act_dim, horizon=2048)
    else:
        raise ValueError('Unknown agent: {}'.format(exp_name))

    sess.run(tf.compat.v1.global_variables_initializer())

    # Params
    total_steps = int(1e6)
    horizon = agent.horizon
    max_ep_len = 10000
    eval_freq = 10

    cprint(f'Running experiment: {exp_name}\n', color='cyan', attrs=['bold'])

    # Start
    start_time = time.time()
    obs = env.reset()
    ep_ret, ep_len = 0, 0
    ep_count = 0
    max_ep_ret = 0

    epochs = total_steps // agent.horizon
    for epoch in range(1, epochs + 1):
        for t in range(1, horizon + 1):
            ac = agent.select_action(obs)

            next_obs, reward, done, _ = env.step(ac)
            ep_ret += reward
            ep_len += 1

            agent.store_transition(obs, ac, reward, done)

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
                      f'training {exp_name}: {epoch/epochs:.1%}')
                # Episode restart
                obs = env.reset()
                ep_ret, ep_len = 0, 0
                ep_count += 1

            if t == horizon:
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0.0 if done else agent.compute_v(obs)
                agent.buffer.finish_path(last_val)
                break

        agent.update()

        # Steps we have trained.
        step = epoch * horizon

        # Evaluate
        if epoch % eval_freq == 0 and args.allow_eval:
            avg_ret, evg_len = evaluate(
                env_eval, agent, num_ep=10, max_ep_len=max_ep_len)
            # Summary
            eval_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag='eval/avg_len', simple_value=evg_len),
                tf.compat.v1.Summary.Value(
                    tag='eval/avg_reward', simple_value=avg_ret)
            ])
            summary_writer.add_summary(eval_summary, step)
            # Save the best weights
            if avg_ret >= max_ep_ret and args.save_model:
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


if __name__ == '__main__':
    main()
