#!/bin/zsh
ALGO=$1
ENV=$2
DIR_NAME=$3

CUDA_VISIBLE_DEVICES=0 python run_deepq_atari.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python run_deepq_atari.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python run_deepq_atari.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python run_deepq_atari.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python run_deepq_atari.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python run_deepq_atari.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
echo 'Running six experiments...'