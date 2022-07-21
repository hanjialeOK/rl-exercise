import os
import gym

from common.monitor import Monitor
from common.vec_env.dummy_vec_env import DummyVecEnv
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.wrappers import ClipActionsWrapper


def make_vec_env(env_id, num_env, seed,
                 env_kwargs=None,
                 start_index=0,
                 monitor_dir=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    env_kwargs = env_kwargs or {}

    def make_env(rank):
        def _init():
            env = gym.make(env_id, **env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(
                rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, allow_early_resets=True)
            # Optionally, wrap the environment with the provided wrapper
            # if isinstance(env.action_space, gym.spaces.Box):
            #     env = ClipActionsWrapper(env)
            return env
        return _init

    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])
