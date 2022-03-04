# RL-exercise

## Introduction

Codes are now mainly copyed from [google/dopamine](https://github.com/google/dopamine). The aim is to learn RL and tensorflow1.

## Usage

### Value-based

```c
usage: run_experiment.py [-h] [--dir_name DIR_NAME]
                         [--exp_name {dqn,ddqn,prior,duel,c51,ddqn+prior,ddqn+duel}]
                         [--env_name ENV_NAME] [--sticky]
                         [--disk_dir DISK_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --dir_name DIR_NAME   Dir name
  --exp_name {dqn,ddqn,prior,duel,c51,ddqn+prior,ddqn+duel}
                        Experiment name
  --env_name ENV_NAME   Env name
  --sticky              Sticky actions
  --disk_dir DISK_DIR   Data disk dir
```

example:

```c
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --exp_name dqn
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --sticky --exp_name c51
```

### Policy-based

```c
usage: run_pg_mujoco.py [-h] [--dir_name DIR_NAME] [--disk_dir DISK_DIR]
                        [--env_name ENV_NAME] [--exp_name {TRPO,PPO,PPO2}]
                        [--allow_eval] [--save_model]

optional arguments:
  -h, --help            show this help message and exit
  --dir_name DIR_NAME   Dir name
  --disk_dir DISK_DIR   Data disk dir
  --env_name ENV_NAME
  --exp_name {TRPO,PPO,PPO2}
                        Experiment name
  --allow_eval          Whether to eval agent
  --save_model          Whether to save model
```

example:

```c
CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --exp_name PPO
CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --exp_name TRPO --allow_eval
```
