#!/bin/zsh
# Usage: zsh run_baselines.sh ppo2 Ant-v2 baselines-PPO 1

ALGO=$1
ENV=$2
DIR_NAME=$3
NUM_ENV=$4

echo "Running ${ALGO} of baselines in ${ENV} for six experiments..."
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=0 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed0 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=1 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed1 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=2 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed2 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=3 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed3 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=4 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed4 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=5 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed5 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=6 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed6 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=7 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed7 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=8 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed8 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
    --alg=${ALGO} --env=${ENV} --num_timesteps=1e7 --num_env=${NUM_ENV} --seed=9 \
    --log_path=/data/hanjl/my_results/atari/${ENV}/${DIR_NAME}/seed9 \
    > /dev/null &
echo "Running six experiments..."