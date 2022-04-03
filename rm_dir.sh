BASE_DIR=$1
TARGET_DIR=$2
ENV=('Ant-v2' 'HalfCheetah-v2' 'Hopper-v2' 'Humanoid-v2' 'HumanoidStandup-v2'
     'InvertedDoublePendulum-v2' 'InvertedPendulum-v2' 'Reacher-v2' 'Swimmer-v2' 'Walker2d-v2')
LEN=${#ENV[*]}

for i in $(seq 1 ${LEN})
do
    cd ${BASE_DIR}"/"${ENV[i]}
    if [ ! -d ${TARGET_DIR} ]; then
        echo "No such dir: ${BASE_DIR}"/"${ENV[i]}"/"${TARGET_DIR}"
    else
        rm -rf ${TARGET_DIR}
        echo "Deleted ${BASE_DIR}"/"${ENV[i]}"/"${TARGET_DIR}."
    fi
done
