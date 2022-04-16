#!/bin/zsh
# Usage: zsh run_baselines.sh ppo2 Ant-v2 baselines-PPO 1

ALGO=$1
ENV=$2
DIR_NAME=$3
NUM_ENV=$4

CUDA_VISIBLE_DEVICES=0 python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e6 --num_env=${NUM_ENV}\
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed0 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e6 --num_env=${NUM_ENV}\
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed1 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e6 --num_env=${NUM_ENV}\
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed2 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e6 --num_env=${NUM_ENV}\
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed3 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e6 --num_env=${NUM_ENV}\
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed4 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e6 --num_env=${NUM_ENV}\
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed5 \
    > /dev/null &
echo "Running six experiments..."