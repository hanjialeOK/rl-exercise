# /bin/zsh
ALGO=$1
ENV=$2
DIR_NAME=$3
DELAY=$4

echo 'Sleeping...'
sleep ${DELAY}
echo 'Wake up!'
CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python run_pg_mujoco.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python run_pg_mujoco.py --alg ${ALGO} --env ${ENV}  --dir_name ${DIR_NAME} > /dev/null &
echo 'Running six experiments...'