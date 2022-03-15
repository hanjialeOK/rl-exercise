# /bin/zsh
ALGO=PPO2
DIR_NAME=PPO2-baselines2

CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --exp_name ${ALGO} --allow_eval --dir_name ${DIR_NAME} > /dev/null &
sleep 2
CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --exp_name ${ALGO} --allow_eval --dir_name ${DIR_NAME} > /dev/null &
sleep 2
CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --exp_name ${ALGO} --allow_eval --dir_name ${DIR_NAME} > /dev/null &
sleep 2
CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --exp_name ${ALGO} --allow_eval --dir_name ${DIR_NAME} > /dev/null &
sleep 2
CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --exp_name ${ALGO} --allow_eval --dir_name ${DIR_NAME} > /dev/null &
sleep 2
CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --exp_name ${ALGO} --allow_eval --dir_name ${DIR_NAME} > /dev/null &
echo 'Running six experiments...'