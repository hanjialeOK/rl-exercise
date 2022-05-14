import numpy as np
import tensorflow as tf
import gym
import time
import os
import argparse
import collections

# from common.cmd_util import make_vec_env
from common.vec_normalize import VecNormalize
# from baselines.common.cmd_util import make_vec_env
# from baselines.common.vec_env.vec_normalize import VecNormalize

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
                        choices=['A2C', 'VPG', 'TRPO', 'TRPO2',
                                 'PPO', 'PPO2', 'PPOV', 'DISC', 'DISC2'],
                        help='Experiment name')
    parser.add_argument('--allow_eval', action='store_true',
                        help='Whether to eval agent')
    parser.add_argument('--save_model', action='store_true',
                        help='Whether to save model')
    parser.add_argument('--total_steps', type=float, default=1e6,
                        help='Total steps trained')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise

    timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    base_name = args.alg + '-' + timestamp
    base_dir = os.path.join(
        args.data_dir, f"my_results/{args.env}/{args.dir_name}/{base_name}")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    total_steps = int(args.total_steps)

    # Dump config.
    # locals() can only be put at main(), instead of __main__.
    # config = convert_json(locals())
    # save_json(config, base_dir)

    # Create dir
    summary_dir = os.path.join(base_dir, "tf1_summary")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    progress_txt = os.path.join(base_dir, 'progress.txt')
    eval_txt = os.path.join(base_dir, 'eval.txt')
    with open(progress_txt, 'w') as f1, open(eval_txt, 'w') as f2:
        f1.write('Step\tAvgEpRet\n')
        f2.write('Step\tAvgEpRet\n')

    # Random seed
    seed = int(time.time()) % 1000
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

    # Environment
    env = gym.make(args.env)
    env.seed(seed)
    env_eval = gym.make(args.env)
    env_eval.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Normalized rew and obs
    env = VecNormalize(env)

    # Tensorboard
    summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

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

    # Hyperparameters in http://arxiv.org/abs/1709.06560
    if args.alg == 'A2C':
        import pg.agents.a2c as A2C
        agent = A2C.A2CAgent(sess, obs_dim, act_dim, horizon=5,
                             gamma=0.995, lam=0.97)
        # 1M // 5 // 488 = 409
        log_interval = total_steps // agent.horizon // 488
    elif args.alg == 'VPG':
        import pg.agents.vpg as VPG
        agent = VPG.VPGAgent(sess, obs_dim, act_dim, horizon=1024,
                             gamma=0.995, lam=0.97)
        # 1M // 2048 / 488 = 1
        log_interval = 2
    elif args.alg == 'TRPO':
        import pg.agents.trpo as TRPO
        agent = TRPO.TRPOAgent(sess, obs_dim, act_dim, horizon=1024,
                               gamma=0.995, lam=0.97, cg_iters=10)
        # 1M // 1024 / 488 = 1
        log_interval = 2
    elif args.alg == 'TRPO2':
        import pg.agents.trpo2 as TRPO2
        agent = TRPO2.TRPOAgent(sess, obs_dim, act_dim, horizon=1024,
                                gamma=0.995, lam=0.97, cg_iters=10)
        # 1M // 1024 / 488 = 1
        log_interval = 2
    elif args.alg == 'PPO':
        import pg.agents.ppo as PPO
        agent = PPO.PPOAgent(sess, obs_dim, act_dim, horizon=2048,
                             gamma=0.995, lam=0.97, fixed_lr=True)
        # 1M // 2048 / 488 = 1
        log_interval = 1
    elif args.alg == 'PPOV':
        raise NotImplementedError
        import pg.agents.ppo2_distv as PPOV
        agent = PPOV.PPOAgent(sess, obs_dim, act_dim, horizon=2048)
        # 1M // 2048 / 488 = 1
        log_interval = 1
    elif args.alg == 'PPO2':
        import pg.agents.ppo2 as PPO2
        agent = PPO2.PPOAgent(sess, summary_writer, obs_dim, act_dim, horizon=2048,
                              gamma=0.995, lam=0.97, fixed_lr=False)
        # 1M // 2048 / 488 = 1
        log_interval = 1
    elif args.alg == 'DISC':
        import pg.agents.disc as DISC
        agent = DISC.PPOAgent(sess, summary_writer, obs_dim, act_dim, horizon=2048,
                              gamma=0.995, lam=0.97, fixed_lr=False)
        # 1M // 2048 / 488 = 1
        log_interval = 1
    elif args.alg == 'DISC2':
        import pg.agents.disc2 as DISC2
        agent = DISC2.PPOAgent(sess, summary_writer, obs_dim, act_dim, horizon=2048,
                               gamma=0.995, lam=0.97, fixed_lr=False)
        # 1M // 2048 / 488 = 1
        log_interval = 1
    else:
        raise ValueError('Unknown agent: {}'.format(args.alg))

    sess.run(tf.compat.v1.global_variables_initializer())

    # Params
    horizon = agent.horizon

    cprint(f'Running experiment...\n'
           f'Env: {args.env}, Alg: {args.alg}\n'
           f'Logging dir: {base_dir}\n',
           color='cyan', attrs=['bold'])

    # Start
    start_time = time.time()
    obs = env.reset()
    # max_ep_ret = 0
    max_ep_len = env.spec.max_episode_steps
    ep_ret_buf = collections.deque(maxlen=100)
    ep_len_buf = collections.deque(maxlen=100)
    ep_len = 0
    ep_ret = 0.

    epochs = total_steps // horizon
    # Openai spinningup implementation
    for epoch in range(1, epochs + 1):
        for t in range(1, horizon + 1):
            ac = agent.select_action(obs)

            next_obs, reward, done, info = env.step(ac)
            ep_len += 1
            ep_ret += env._unnormalize_ret(reward)

            done = done if ep_len < max_ep_len else False
            agent.store_transition(obs, ac, reward, done)

            obs = next_obs

            terminal = done or ep_len == max_ep_len
            if terminal or t == horizon:
                last_val = 0. if done else agent.compute_v(next_obs)
                agent.buffer.finish_path(last_val)
                if terminal:
                    ep_ret_buf.append(ep_ret)
                    ep_len_buf.append(ep_len)
                obs = env.reset()
                ep_len = 0
                ep_ret = 0.

        # Progress ratio
        frac = 1.0 - (epoch - 1.0) / epochs
        # Steps we have reached.
        step = epoch * horizon
        # Log to board
        log2board = epoch % log_interval == 0

        [pi_loss, vf_loss, ent, kl] = agent.update(frac, log2board, step)

        [rms_obs, rms_ret] = agent.buffer.get_rms_data()
        env.update_rms(obs=rms_obs, ret=rms_ret)
        agent.buffer.reset()

        if epoch % log_interval == 0:
            avg_ep_ret = np.mean(ep_ret_buf)
            avg_ep_len = np.mean(ep_len_buf)

            # Episode summary
            train_summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(
                    tag="train/avglen", simple_value=avg_ep_len),
                tf.compat.v1.Summary.Value(
                    tag="train/avgret", simple_value=avg_ep_ret),
                tf.compat.v1.Summary.Value(
                    tag="loss/avgpiloss", simple_value=pi_loss),
                tf.compat.v1.Summary.Value(
                    tag="loss/avgvloss", simple_value=vf_loss),
                tf.compat.v1.Summary.Value(
                    tag="loss/avgentropy", simple_value=ent),
                tf.compat.v1.Summary.Value(
                    tag="loss/avgkl", simple_value=kl)
            ])
            summary_writer.add_summary(train_summary, step)

            log_infos = []
            log_infos.append(f'Env: {args.env} | Alg: {args.alg} | '
                             f'TotalSteps: {total_steps:.1e} | '
                             f'Horizon: {horizon}')
            log_infos.append(f'Epoch: {epoch}/{epochs}: {epoch/epochs:.1%} | '
                             f'AvgLen: {avg_ep_len:.1f} | '
                             f'AvgRet: {avg_ep_ret:.1f}')
            log_infos.append(f'pi_loss: {pi_loss:.4f} | '
                             f'v_loss: {vf_loss:.4f} | '
                             f'entropy: {ent:.4f} | '
                             f'kl: {kl:.4f}')
            info_lens = [len(info) for info in log_infos]
            max_info_len = max(info_lens)
            n_slashes = max_info_len + 2
            print("+" + "-"*n_slashes + "+")
            for info in log_infos:
                print(f"| {info:{max_info_len}s} |")
            print("+" + "-"*n_slashes + "+")

            with open(progress_txt, 'a') as f:
                f.write(f"{step}\t{avg_ep_ret}\n")

        # Evaluate
        if epoch % 100 == 0 and args.allow_eval:
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
