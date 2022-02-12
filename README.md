# RL-exercise

## Introduction

Codes are now mainly copyed from [google/dopamine](https://github.com/google/dopamine). The aim is to learn RL and tensorflow1.

## Usage

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