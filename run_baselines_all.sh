#!/bin/zsh
# Usage: zsh run_baselines_all.sh ppo2 baselines-PPO

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
DIR_NAME=$2
ENV=('Ant-v2' 'HalfCheetah-v2' 'Hopper-v2' 'Humanoid-v2' 'HumanoidStandup-v2'
     'InvertedDoublePendulum-v2' 'InvertedPendulum-v2' 'Reacher-v2' 'Swimmer-v2' 'Walker2d-v2')
LEN=${#ENV[*]}
SECONDS=0

echo "Running ${ALGO} of baselines..."
echo "=================================================="

for i in $(seq 1 ${LEN})
do
    echo "${CYAN}${BOLD}Running ${ENV[i]} (${i}/${LEN}) for six experiments...${RESET}"
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=0 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed0 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=1 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed1 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=2 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed2 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=3 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed3 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=4 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed4 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=5 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed5 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=6 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed6 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=7 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed7 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=8 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed8 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV[i]} --num_timesteps=1e6 --num_env=1 --seed=9 \
        --log_path=/data/hanjl/my_results/${ENV[i]}/${DIR_NAME}/seed9 \
        > /dev/null &
    # Waiting for all subprocess finished.
    wait
    sleep 10
done

duration=${SECONDS}
h=$[${duration}/3600]
m=$[(${duration}/60)%60]
s=$[${duration}%60]
printf "%s%02d:%02d:%02d%s\\n" "Completed! Time taken: " "${h}" "${m}" "${s}" "."
