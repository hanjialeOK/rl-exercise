import numpy as np
import tensorflow as tf
import gym
import time
import os
import argparse
import collections

import pg.agents.vpg as VPG
import pg.agents.trpo as TRPO
import pg.agents.ppo as PPO
import pg.agents.ppo2 as PPO2
import pg.agents.ppo_m as PPOM
import common.vec_normalize as Wrapper

from termcolor import cprint, colored
from common.serialization_utils import convert_json, save_json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(env_eval, agent, n_eval_episodes=10):
    ret_sum = 0.
    len_sum = 0

    for i in range(1, n_eval_episodes + 1):
        obs = env_eval.reset()
        ep_ret = 0.
        ep_len = 0
        while True:
            ac = agent.select_action(obs, deterministic=False)
            obs, reward, done, _ = env_eval.step(ac)
            # if isinstance(obs, np.ndarray):
            #     obs = obs[0]
            #     reward = reward[0]
            #     done = done[0]
            ep_ret += reward
            ep_len += 1
            if done:
                cprint(f'\rEvaluate: {i}/{n_eval_episodes}',
                       color='cyan', attrs=['bold'], end='')
                ret_sum += ep_ret
                len_sum += ep_len
                break
    avg_eval_ret = ret_sum / n_eval_episodes
    avg_eval_len = len_sum / n_eval_episodes
    cprint(f'\navg_eval_len: {ep_len}, avg_eval_ret: {ep_ret:.1f}',
           color='cyan', attrs=['bold'])

    return avg_eval_ret, avg_eval_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str,
                        default='default', help='Dir name')
    parser.add_argument('--data_dir', type=str, default='/data/hanjl',
                        help='Data disk dir')
    parser.add_argument('--env', type=str,
                        default='Walker2d-v2')
    parser.add_argument('--alg', type=str, default='PPO',
                        choices=['VPG', 'TRPO', 'PPO', 'PPO2', 'PPOM'],
                        help='Experiment name',)
    parser.add_argument('--allow_eval', action='store_true',
                        help='Whether to eval agent')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model')
    parser.add_argument('--total_steps', type=int, default=1e6,
                        help='Total steps trained')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise

    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    base_name = args.alg + '-' + timestamp
    dir_name = args.dir_name
    exp_name = args.alg
    env_name = args.env
    base_dir = os.path.join(
        args.data_dir, f"my_results/{env_name}/{dir_name}/{base_name}")
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
    train_txt = os.path.join(base_dir, 'train.txt')
    eval_txt = os.path.join(base_dir, 'eval.txt')
    with open(train_txt, 'w') as f1, open(eval_txt, 'w') as f2:
        f1.write('Step\tAvgLength\tAvgReturn\n')
        f2.write('Step\tAvgReturn\n')

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
    max_action = float(env.action_space.high[0])

    # Normalized rew and obs
    env = Wrapper.VecNormalize(env)
    env_eval = Wrapper.VecNormalize(env_eval, ret=False)
    # env = make_vec_env(env_name, 'mujoco', 1, seed,
    #                    reward_scale=1.0, flatten_dict_observations=True)
    # env = VecNormalize(env, use_tf=False)

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
    gpu_list = tf.compat.v1.config.experimental.list_physical_devices('GPU')
    cprint(f'Tensorflow version: {tf.__version__}, '
           f'GPU available: {tf.test.is_gpu_available()}, '
           f'GPU count: {len(gpu_list)}\n'
           f'{gpu_list}\n',
           color='cyan', attrs=['bold'])

    if args.alg == 'VPG':
        agent = VPG.VPGAgent(sess, obs_dim, act_dim, horizon=1000)
    elif args.alg == 'TRPO':
        agent = TRPO.TRPOAgent(sess, obs_dim, act_dim, horizon=1000)
    elif args.alg == 'PPO':
        agent = PPO.PPOAgent(sess, obs_dim, act_dim, horizon=2048,
                             summary_writer=summary_writer)
    elif args.alg == 'PPOM':
        agent = PPOM.PPOAgent(sess, obs_dim, act_dim, horizon=1000)
    elif args.alg == 'PPO2':
        agent = PPO2.PPOAgent(sess, obs_dim, act_dim, horizon=2048)
    else:
        raise ValueError('Unknown agent: {}'.format(args.alg))

    sess.run(tf.compat.v1.global_variables_initializer())

    # Params
    total_steps = int(args.total_steps)
    horizon = agent.horizon
    eval_freq = 10

    cprint(f'Running experiment: {args.alg}\n', color='cyan', attrs=['bold'])

    # Start
    start_time = time.time()
    obs = env.reset()
    ep_ret, ep_len = 0.0, 0
    ep_count = 0
    max_ep_ret = 0
    ep_ret_buf = collections.deque(maxlen=100)
    ep_len_buf = collections.deque(maxlen=100)

    epochs = total_steps // agent.horizon
    for epoch in range(1, epochs + 1):
        # obs = env.reset()
        # ep_ret, ep_len = 0.0, 0
        for t in range(1, horizon + 1):
            ac = agent.select_action(obs)

            next_obs, reward, done, info = env.step(ac)
            if isinstance(done, np.ndarray):
                next_obs = next_obs[0]
                reward = reward[0]
                done = done[0]
                info = info[0]

            # ep_ret += reward
            # ep_len += 1

            agent.store_transition(obs, ac, reward, done)

            obs = next_obs

            if done:
                ep_count += 1
                ep_ret = info['episode']['r']
                ep_len = info['episode']['l']
                ep_ret_buf.append(ep_ret)
                ep_len_buf.append(ep_len)
                # Episode restart
                obs = env.reset()
                ep_ret, ep_len = 0.0, 0

        # If trajectory didn't reach terminal state, bootstrap value target
        last_val = agent.compute_v(obs)
        agent.buffer.finish_path(last_val)

        frac = 1.0 - (epoch - 1.0) / epochs

        pi_loss, v_loss, entropy, kl = agent.update(frac)

        # Steps we have trained.
        step = epoch * horizon

        avg_ep_ret = np.mean(ep_ret_buf)
        avg_ep_len = np.mean(ep_len_buf)

        # Episode summary
        train_summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
                tag="train/avglen", simple_value=avg_ep_len),
            tf.compat.v1.Summary.Value(
                tag="train/avgret", simple_value=avg_ep_ret),
            tf.compat.v1.Summary.Value(
                tag="loss/avgkl", simple_value=kl),
            tf.compat.v1.Summary.Value(
                tag="loss/avgpiloss", simple_value=pi_loss),
            tf.compat.v1.Summary.Value(
                tag="loss/avgvloss", simple_value=v_loss),
            tf.compat.v1.Summary.Value(
                tag="loss/avgentropy", simple_value=entropy)
        ])
        summary_writer.add_summary(train_summary, step)
        with open(train_txt, 'a') as f:
            f.write(f"{step}\t{avg_ep_len}\t{avg_ep_ret}\n")
        ep_ret_text = colored(f'{avg_ep_ret:.1f}',
                              color='green', attrs=['bold'])
        print(f'@Epoch: {epoch}/{epochs}, '
              f'AvgLen: {avg_ep_len}, AvgRet: {ep_ret_text}, '
              f'Algo {args.alg}: {epoch/epochs:.1%}\n'
              f'pi_loss: {pi_loss:.4f}, v_loss: {v_loss:.4f}, '
              f'entropy: {entropy:.4f}, kl: {kl:.4f}')

        # Evaluate
        if epoch % eval_freq == 0 and args.allow_eval:
            avg_ret, evg_len = evaluate(env_eval, agent)
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
            with open(eval_txt, 'a') as f:
                f.write(f"{step}\t{avg_ret}\n")

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
