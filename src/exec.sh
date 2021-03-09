#!/bin/bash

##############
#  settings  #
##############

SCRIPT="baseline"
DATASET_NAME="mnist"
EPOCHS=20
PROJ_MODE="channels"
N_JOBS=10

ATTACK_LIBRARY="art"
DEVICE="cpu"
LOAD="False"
DEBUG="False"

#######
# run #
#######

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="../experiments/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_${SCRIPT}_${DATASET_NAME}_${EPOCHS}_${ATTACK_LIBRARY}_out.txt"


if [ $SCRIPT = "baseline" ]; then
    python3 "train_attack_baseline.py" --dataset_name=$DATASET_NAME --epochs=$EPOCHS --load=$LOAD --debug=$DEBUG \
                                        --attack_library=$ATTACK_LIBRARY > $OUT
elif [ $SCRIPT = "random_ensemble" ]; then
    python3 "train_attack_random_ensemble.py" --dataset_name=$DATASET_NAME --epochs=$EPOCHS  --load=$LOAD --debug=$DEBUG \
                                        --projection_mode=$PROJ_MODE --attack_library=$ATTACK_LIBRARY > $OUT
elif [ $SCRIPT = "parallel_random_ensemble" ]; then
    python3 "train_attack_parallel_random_ensemble.py" --dataset_name=$DATASET_NAME --epochs=$EPOCHS  --load=$LOAD \
                                        --debug=$DEBUG --projection_mode=$PROJ_MODE --attack_library=$ATTACK_LIBRARY \
                                        --n_jobs=$N_JOBS > $OUT
fi 

deactivate