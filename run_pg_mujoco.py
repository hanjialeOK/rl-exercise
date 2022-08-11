import numpy as np
import tensorflow as tf
import gym
import time
import random
import os
import argparse
import collections
import pickle

from common.cmd_util import make_vec_env
from common.vec_env.vec_normalize import VecNormalize
# from common.vec_normalize import VecNormalize2
# from baselines.common.cmd_util import make_vec_env
# from baselines.common.vec_env.vec_normalize import VecNormalize
from common.logger import Logger

from termcolor import cprint, colored

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
    parser.add_argument('--alg', type=str, default='PPO2',
                        choices=['A2C', 'ACER', 'VPG', 'TRPO', 'DISDC',
                                 'PPO', 'PPO2', 'DISC', 'GePPO', 'GeDISC'],
                        help='Experiment name')
    parser.add_argument('--allow_eval', action='store_true',
                        help='Whether to eval agent')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model')
    parser.add_argument('--total_steps', type=float, default=1e6,
                        help='Total steps trained')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--uniform', action='store_true',
                        help='Total steps trained')
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
    obs_shape = env.observation_space.shape
    ac_shape = env.action_space.shape

    # Our own simple warpper
    # env = VecNormalize2(env)
    # Openai baselines
    env = make_vec_env(args.env, num_env=1, seed=args.seed)
    env = VecNormalize(env)

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False))
    # Fix save_weights and load_weights for keras.
    # tf.compat.v1.keras.backend.set_session(sess)

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

    # Hyperparameters in http://arxiv.org/abs/1709.06560
    if args.alg == 'A2C':
        import pg.agents.a2c as A2C
        agent = A2C.A2CAgent(sess, obs_shape, ac_shape, horizon=5,
                             gamma=0.995, lam=0.97)
    if args.alg == 'ACER':
        import pg.agents.acer as ACER
        agent = ACER.ACERAgent(sess, env, horizon=50, gamma=0.99)
    elif args.alg == 'VPG':
        import pg.agents.vpg as VPG
        agent = VPG.VPGAgent(sess, obs_shape, ac_shape, horizon=1024,
                             gamma=0.995, lam=0.97)
    elif args.alg == 'TRPO':
        import pg.agents.trpo as TRPO
        agent = TRPO.TRPOAgent(sess, obs_shape, ac_shape, horizon=1024,
                               gamma=0.995, lam=0.97, cg_iters=10)
    elif args.alg == 'PPO':
        import pg.agents.ppo as PPO
        agent = PPO.PPOAgent(sess, env, horizon=2048,
                             gamma=0.99, lam=0.95, fixed_lr=False)
    elif args.alg == 'PPO2':
        import pg.agents.ppo2 as PPO2
        agent = PPO2.PPOAgent(sess, env, horizon=2048,
                              gamma=0.99, lam=0.95, fixed_lr=False)
    elif args.alg == 'DISC':
        import pg.agents.disc as DISC
        agent = DISC.PPOAgent(sess, env, horizon=2048,
                              gamma=0.99, lam=0.95, fixed_lr=False)
    elif args.alg == 'DISDC':
        import pg.agents.disdc as DISDC
        agent = DISDC.PPOAgent(sess, env, horizon=2048,
                              gamma=0.99, lam=0.95, fixed_lr=False)
    elif args.alg == 'GeDISC':
        import pg.agents.gedisc as GeDISC
        agent = GeDISC.PPOAgent(sess, env, horizon=2048,
                                gamma=0.99, lam=0.95, fixed_lr=False, uniform=True)
    elif args.alg == 'GePPO':
        import pg.agents.geppo as GePPO
        agent = GePPO.PPOAgent(sess, env, horizon=1024,
                               gamma=0.99, lam=0.95, fixed_lr=False, uniform=False)
    else:
        raise ValueError('Unknown agent: {}'.format(args.alg))

    # 1M // 2048 / 488 = 1
    log_interval = total_steps // agent.horizon // 488

    # Actually, we won't use save_weights because it does not contain logstd.
    tf.compat.v1.keras.backend.set_session(sess)
    sess.run(tf.compat.v1.global_variables_initializer())
    if args.alg == 'ACER':
        sess.run(agent.init_avg_op)

    # with open('/data/hanjl/debug_data4/actor_param.pkl', 'rb') as f:
    #     actor_param = pickle.load(f)
    #     agent.assign_actor_weights(actor_param)
    # with open('/data/hanjl/debug_data4/critic_param.pkl', 'rb') as f:
    #     critic_param = pickle.load(f)
    #     agent.assign_critic_weights(critic_param)

    # Params
    horizon = agent.horizon

    cprint(f'Running experiment...\n'
           f'Env: {args.env}, Alg: {args.alg}\n'
           f'Logging dir: {base_dir}\n',
           color='cyan', attrs=['bold'])

    # Start
    start_time = time.time()
    obs = env.reset()
    raw_obs, _ = env.get_raw()
    # max_ep_ret = 0
    max_ep_len = env.spec.max_episode_steps
    ep_ret_buf = collections.deque(maxlen=100)
    ep_len_buf = collections.deque(maxlen=100)
    ep_len = 0
    ep_ret = 0.

    nupdates = total_steps // horizon
    # Openai spinningup implementation
    for update in range(0, nupdates + 1):
        # Clear buffer
        for t in range(1, horizon + 1):
            ac, val, neglogp, mean, logstd = agent.select_action(obs)

            next_obs, reward, done, info = env.step(ac)
            next_raw_obs, raw_rew = env.get_raw()
            ep_len += 1
            ep_ret += raw_rew

            # done = done if ep_len < max_ep_len else False
            agent.buffer.store(obs=obs, ac=ac, rew=reward, done=done, obs2=next_obs,
                               val=val, neglogp=neglogp, mean=mean, logstd=logstd,
                               raw_obs=raw_obs, raw_rew=raw_rew, raw_obs2=next_raw_obs)

            obs = next_obs
            raw_obs = next_raw_obs

            terminal = done or ep_len == max_ep_len
            if terminal or t == horizon:
                agent.buffer.finish_path()
                if terminal:
                    # obs = env.reset()
                    # raw_obs, _ = env.get_raw()
                    ep_ret_buf.append(ep_ret)
                    ep_len_buf.append(ep_len)
                    ep_len = 0
                    ep_ret = 0.

        # Update rms for env. Uncomment this if using your own wrapper.
        # [rms_obs, rms_ret] = agent.buffer.get_rms_data()
        # env.update_rms(obs=rms_obs, ret=rms_ret)

        if update == 0:
            ep_ret_buf.clear()
            ep_len_buf.clear()
            agent.buffer.reset()
            print('Initialized RMS for env.')
            continue

        # Progress ratio
        frac = 1.0 - (update - 1.0) / nupdates
        # Steps we have reached.
        step = update * horizon

        agent.update(frac, logger)

        if update % log_interval == 0:
            avg_ep_ret = np.mean(ep_ret_buf)
            avg_ep_len = np.mean(ep_len_buf)

            logger.logkv("train/nupdates", update)
            logger.logkv("train/timesteps", step)
            logger.logkv("train/avgeplen", avg_ep_len)
            logger.logkv("train/avgepret", avg_ep_ret)

            logger.loginfo('env', args.env)
            logger.loginfo('alg', args.alg)
            logger.loginfo('totalsteps', total_steps)
            logger.loginfo('totalupdates', nupdates)
            logger.loginfo('horizon', horizon)
            logger.loginfo('progress', f'{update/nupdates:.1%}')

            logger.dumpkvs(timestep=step)

        # Evaluate
        if update % 100 == 0 and args.allow_eval:
            raise NotImplementedError
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
                agent.bundle(checkpoint_dir, update)
                max_ep_ret = avg_ret
            # Log data
            with open(eval_txt, 'a') as f:
                f.write(f"{step}\t{avg_ret}\n")

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
