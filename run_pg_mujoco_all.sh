#!/bin/zsh
ALGO=$1
DIR_NAME=$2
ENV=('Ant-v2' 'HalfCheetah-v2' 'Hopper-v2' 'Humanoid-v2' 'HumanoidStandup-v2'
     'InvertedDoublePendulum-v2' 'InvertedPendulum-v2' 'Reacher-v2' 'Swimmer-v2' 'Walker2d-v2')
LEN=${#ENV[*]}

for i in $(seq 1 ${LEN})
do
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
    echo "Running ${ENV[i]} (${i}/${LEN}) for six experiments..."
    sleep 30m
    while true
    do
        pid0=$(fuser /dev/nvidia0)
        pid1=$(fuser /dev/nvidia1)
        if [ ! ${pid0} ] && [ ! ${pid1} ]; then
            echo "Completed!"
            break
        else
            echo "Busy! ${pid0}${pid1} is running."
            sleep 1m
        fi
    done
done