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
STEPS=$3
ENV=(
     'Alien' 'Amidar' 'Assault' 'Asterix' 'Asteroids' 'Atlantis' 'BankHeist' 'BattleZone' 'BeamRider' 'Bowling'
     'Boxing' 'Breakout' 'Centipede' 'ChopperCommand' 'CrazyClimber' 'DemonAttack' 'DoubleDunk' 'Enduro' 'FishingDerby' 'Freeway'
     'Frostbite' 'Gopher' 'Gravitar' 'IceHockey' 'Jamesbond' 'Kangaroo' 'Krull' 'KungFuMaster' 'MontezumaRevenge' 'MsPacman'
     'NameThisGame' 'Pitfall' 'Pong' 'PrivateEye' 'Qbert' 'Riverraid' 'RoadRunner' 'Robotank' 'Seaquest' 'SpaceInvaders'
     'StarGunner' 'Tennis' 'TimePilot' 'Tutankham' 'UpNDown' 'Venture' 'VideoPinball' 'WizardOfWor' 'Zaxxon'
     )
VERSION='NoFrameskip-v4'
LEN=${#ENV[*]}
SECONDS=0

echo "Running ${ALGO} of baselines..."
echo "=================================================="

for i in $(seq 1 ${LEN})
do
    echo "${CYAN}${BOLD}Running ${ENV[i]} (${i}/${LEN}) for six experiments...${RESET}"
    ENV_NAME=${ENV[i]}${VERSION}
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=0 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed0 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=1 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed1 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=2 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed2 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=3 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed3 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=4 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed4 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=5 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed5 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=6 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed6 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=7 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed7 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=8 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed8 \
        > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python -m baselines.run \
        --alg=${ALGO} --env=${ENV_NAME} --num_timesteps=${STEPS} --num_env=8 --seed=9 \
        --log_path=/data/hanjl/my_results/atari/${ENV_NAME}/${DIR_NAME}/seed9 \
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
