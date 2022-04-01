# /bin/zsh
ALGO=$1
ENV=$2
DIR_NAME=$3
DELAY=$4

echo 'Sleeping...'
sleep ${DELAY}
echo 'Wake up!'
CUDA_VISIBLE_DEVICES=0 python -m baselines.run \
    --alg=ppo2 --env=${ENV} --num_timesteps=1e6 \
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed0 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python -m baselines.run \
    --alg=ppo2 --env=${ENV} --num_timesteps=1e6 \
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed1 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=0 python -m baselines.run \
    --alg=ppo2 --env=${ENV} --num_timesteps=1e6 \
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed2 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python -m baselines.run \
    --alg=ppo2 --env=${ENV} --num_timesteps=1e6 \
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed3 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python -m baselines.run \
    --alg=ppo2 --env=${ENV} --num_timesteps=1e6 \
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed4 \
    > /dev/null &
sleep 5
CUDA_VISIBLE_DEVICES=1 python -m baselines.run \
    --alg=ppo2 --env=${ENV} --num_timesteps=1e6 \
    --log_path=/data/hanjl/my_results/${ENV}/${DIR_NAME}/seed5 \
    > /dev/null &
echo 'Running six experiments...'