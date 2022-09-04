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
DIR_NAME=$2
STEPS=$3
# default: 0
GPU0="${4:=0}"
# default: 1
GPU1="${5:=1}"
ENV=('Alien' 'Amidar' 'Assault' 'Asterix' 'Asteroids' 'Atlantis' 'BankHeist' 'BattleZone' 'BeamRider' 'Bowling'
     'Boxing' 'Breakout' 'Centipede' 'ChopperCommand' 'CrazyClimber' 'DemonAttack' 'DoubleDunk' 'Enduro' 'FishingDerby' 'Freeway'
     'Frostbite' 'Gopher' 'Gravitar' 'IceHockey' 'Jamesbond' 'Kangaroo' 'Krull' 'KungFuMaster' 'MontezumaRevenge' 'MsPacman'
     'NameThisGame' 'Pitfall' 'Pong' 'PrivateEye' 'Qbert' 'Riverraid' 'RoadRunner' 'Robotank' 'Seaquest' 'SpaceInvaders'
     'StarGunner' 'Tennis' 'TimePilot' 'Tutankham' 'UpNDown' 'Venture' 'VideoPinball' 'WizardOfWor' 'Zaxxon')
VERSION='NoFrameskip-v4'
LEN=${#ENV[*]}
SECONDS=0

echo "Running ${BOLD}${ALGO}${RESET} of rl-exercise..."
echo "=================================================="

for i in $(seq 1 ${LEN})
do
    echo "${CYAN}${BOLD}Running ${ENV[i]} (${i}/${LEN}) for six experiments...${RESET}"
    CUDA_VISIBLE_DEVICES=${GPU0} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 0 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU0} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 1 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU0} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 2 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU0} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 3 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU0} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 4 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU1} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 5 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU1} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 6 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU1} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 7 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU1} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 8 > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=${GPU1} PYTHONWARNINGS=ignore python run_pg_atari.py \
        --alg ${ALGO} --env ${ENV[i]}${VERSION} --dir_name ${DIR_NAME} --total_steps ${STEPS} --seed 9 > /dev/null &
    # Waiting for all subprocess finished.
    wait
    sleep 10
done

duration=${SECONDS}
h=$[${duration}/3600]
m=$[(${duration}/60)%60]
s=$[${duration}%60]
echo "Completed! Time taken: ${h}h:${m}m:${s}s."