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
ENV=('Ant-v2' 'HalfCheetah-v2' 'Hopper-v2' 'Humanoid-v2' 'HumanoidStandup-v2'
     'InvertedDoublePendulum-v2' 'InvertedPendulum-v2' 'Reacher-v2' 'Swimmer-v2' 'Walker2d-v2')
LEN=${#ENV[*]}
SECONDS=0

echo "Running rl-exercise..."
echo "=================================================="

for i in $(seq 1 ${LEN})
do
    echo "${CYAN}${BOLD}Running ${ENV[i]} (${i}/${LEN}) for six experiments...${RESET}"
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
        --alg ${ALGO} --env ${ENV[i]} --dir_name ${DIR_NAME} > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
        --alg ${ALGO} --env ${ENV[i]} --dir_name ${DIR_NAME} > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
        --alg ${ALGO} --env ${ENV[i]} --dir_name ${DIR_NAME} > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
        --alg ${ALGO} --env ${ENV[i]} --dir_name ${DIR_NAME} > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
        --alg ${ALGO} --env ${ENV[i]} --dir_name ${DIR_NAME} > /dev/null &
    sleep 5
    CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore python run_pg_mujoco.py \
        --alg ${ALGO} --env ${ENV[i]} --dir_name ${DIR_NAME} > /dev/null &
    # Waiting for all subprocess finished.
    wait
    sleep 10
done

duration=${SECONDS}
h=$[${duration}/3600]
m=$[(${duration}/60)%60]
s=$[${duration}%60]
printf "%s%02d:%02d:%02d%s\\n" "Completed! Time taken: " "${h}" "${m}" "${s}" "."
