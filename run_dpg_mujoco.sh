#!/bin/zsh

# color
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
BLUE=$(tput setaf 4)
MAGENTA=$(tput setaf 5)
CYAN=$(tput setaf 6)
WHIHE=$(tput setaf 7)
# mode
BOLD=$(tput bold)
RESET=$(tput sgr0)

ALGO=$1
ENV=$2
DIR_NAME=$3
STEPS=$4
# default: 0
GPU0="${5:=0}"
# default: 1
GPU1="${6:=1}"
SECONDS=0

echo "Running ${BOLD}${ALGO}${RESET} of rl-exercise in ${BOLD}${ENV}${RESET} for 4 experiments..."
CUDA_VISIBLE_DEVICES=${GPU0} PYTHONWARNINGS=ignore python run_dpg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 0 > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=${GPU0} PYTHONWARNINGS=ignore python run_dpg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 1 > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=${GPU1} PYTHONWARNINGS=ignore python run_dpg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 2 > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=${GPU1} PYTHONWARNINGS=ignore python run_dpg_mujoco.py \
    --alg ${ALGO} --env ${ENV} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 3 > /dev/null &
wait

duration=${SECONDS}
h=$[${duration}/3600]
m=$[(${duration}/60)%60]
s=$[${duration}%60]
echo "Completed! Time taken: ${h}h:${m}m:${s}s."