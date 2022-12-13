import numpy as np
import tensorflow as tf
import gym
import time
import os
import argparse
import random
import collections

from termcolor import cprint, colored
from common.logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(env, agent, n_episodes=5):
    eval_epinfobuf = []

    for i in range(1, n_episodes + 1):
        obs = env.reset()
        ep_ret = 0.
        ep_len = 0
        while True:
            ac = agent.select_action(obs, noise=False)
            obs, reward, done, _ = env.step(ac)
            ep_ret += reward
            ep_len += 1
            if done:
                eval_epinfobuf.append({'r': ep_ret, 'l': ep_len})
                break

    return eval_epinfobuf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='default', help='Dir name')
    parser.add_argument('--data_dir', type=str, default='/data/hanjl',
                        help='Data disk dir')
    parser.add_argument('--env', type=str,
                        default='HalfCheetah-v2')
    parser.add_argument('--alg', type=str, default='DDPG',
                        choices=['DPG', 'DDPG', 'TD3', 'SAC'],
                        help='Experiment name',)
    parser.add_argument('--allow_eval', action='store_true',
                        help='Whether to eval agent')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model')
    parser.add_argument('--total_steps', type=float, default=1e6,
                        help='Total steps trained')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise

    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    base_name = args.alg + '_' + f'seed{args.seed}' + '_' + timestamp
    base_dir = os.path.join(
        args.data_dir, f"my_results/{args.env}/{args.dir_name}/{base_name}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    total_steps = int(args.total_steps)

    # Create dir
    summary_dir = os.path.join(base_dir, "tf1_summary")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    progress_csv = os.path.join(base_dir, 'progress.csv')
    log_txt = os.path.join(base_dir, 'log.txt')
    summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)
    logger = Logger(progress_csv, log_txt, summary_writer)

    # Random seed
    tf.compat.v1.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Environment
    env = gym.make(args.env)
    env.seed(args.seed)
    env_eval = gym.make(args.env)
    env_eval.seed(args.seed)
    obs_dim = env.observation_space.shape
    ac_dim = env.action_space.shape
    max_action = float(env.action_space.high[0])

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

    if args.alg == 'DPG':
        raise NotImplementedError
    elif args.alg == 'DDPG':
        import dpg.agents.ddpg as DDPG
        agent = DDPG.DDPGAgent(sess, obs_dim, ac_dim,
                               max_action, noise_scale=0.1)
    elif args.alg == 'TD3':
        import dpg.agents.td3 as TD3
        agent = TD3.TD3Agent(sess, obs_dim, ac_dim,
                             max_action, noise_scale=0.1)
    elif args.alg == 'SAC':
        import dpg.agents.sac as SAC
        agent = SAC.SACAgent(sess, obs_dim, ac_dim, max_action)
    else:
        raise ValueError(f'Unknown agent: {args.alg}')

    tf.compat.v1.keras.backend.set_session(sess)
    sess.run(tf.compat.v1.global_variables_initializer())
    agent.target_params_init()

    # Params
    start_steps = int(25e3)
    log_interval = 4096
    eval_freq = int(5e3)

    cprint(f'Running experiment...\n'
           f'Env: {args.env}, Alg: {args.alg}\n'
           f'Logging dir: {base_dir}\n',
           color='cyan', attrs=['bold'])

    # Start
    epinfobuf = collections.deque(maxlen=100)
    start_time = time.time()
    obs = env.reset()
    ep_ret, ep_len = 0, 0
    ep_count = 0
    max_ep_len = env.spec.max_episode_steps

    for t in range(1, total_steps + 1):
        if t >= start_steps:
            ac = agent.select_action(obs, noise=True)
        else:
            ac = env.action_space.sample()

        next_obs, reward, done, _ = env.step(ac)
        ep_ret += reward
        ep_len += 1

        done = False if ep_len == max_ep_len else done
        agent.buffer.store(obs, next_obs, ac, reward, done)

        obs = next_obs

        if done or ep_len == max_ep_len:
            epinfobuf.append({'r': ep_ret, 'l': ep_len})
            obs = env.reset()
            ep_ret = 0.
            ep_len = 0

        if t >= start_steps:
            agent.update(logger)

        if t % log_interval == 0:
            avg_ep_ret = np.mean([epinfo['r'] for epinfo in epinfobuf])
            avg_ep_len = np.mean([epinfo['l'] for epinfo in epinfobuf])

            logger.logkv("train/timesteps", t)
            logger.logkv("train/avgeplen", avg_ep_len)
            logger.logkv("train/avgepret", avg_ep_ret)

            logger.loginfo('env', args.env)
            logger.loginfo('alg', args.alg)
            logger.loginfo('totalsteps', total_steps)
            logger.loginfo('progress', f'{t/total_steps:.1%}')

            # Evaluate
            eval_epinfobuf = evaluate(env_eval, agent)
            logger.logkv('eval/avgeplen', np.mean([epinfo['l'] for epinfo in eval_epinfobuf]))
            logger.logkv('eval/avgepret', np.mean([epinfo['r'] for epinfo in eval_epinfobuf]))
            # Save the best weights
            # if avg_ret >= max_ep_ret and args.save_model:
            #     print(f'Saving weights into {checkpoint_dir}')
            #     agent.bundle(checkpoint_dir, update)
            #     max_ep_ret = avg_ret

            logger.dumpkvs(timestep=t)

    time_delta = int(time.time() - start_time)
    m, s = divmod(time_delta, 60)
    h, m = divmod(m, 60)
    print(f'Time taken: {h:d}:{m:02d}:{s:02d}')
    print(f"Results saved into {base_dir}")
    logger.close()
    env.close()
    env_eval.close()


if __name__ == '__main__':
    main()
