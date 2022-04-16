#!/bin/zsh
ALGO=$1
ENV=$2
DIR_NAME=$3
NUM_ENV=$4

echo "Running ${ALGO} of rl-exercise in ${ENV} for six experiments..."
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --num_env ${NUM_ENV} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --num_env ${NUM_ENV} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --num_env ${NUM_ENV} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --num_env ${NUM_ENV} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --num_env ${NUM_ENV} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --num_env ${NUM_ENV} > /dev/null &