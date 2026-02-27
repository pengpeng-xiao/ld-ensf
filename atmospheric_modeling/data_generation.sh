#!/bin/bash
# generate data
IC_INDEX=1
IC_DIR="./atmospheric_modeling/data_stubs/"
OUTPUT_DIR="./atmospheric_modeling/data_parameter/"
N_TRAJ=200

trap "echo 'Interrupted. Exiting loop.'; exit 1" SIGINT

for ((i=0; i<$N_TRAJ; i++))
do
    HF0=$(python -c "import random; print(round(random.uniform(0.1, 30.0), 3))")
    SIGMA=$(python -c "import random; import math; print(round(random.uniform(1.0, 4.0), 3))")
    
    echo "[$i] Running with hf0=${HF0}, sigma=${SIGMA}"
    
    mpiexec -n 64 python gen_SWE_force.py \
        --index $IC_INDEX \
        --ic_dir $IC_DIR \
        --output_dir $OUTPUT_DIR \
        --hf0 $HF0 \
        --sigma $SIGMA
done

# shuffle data into train/valid/test
SRC_DIR="./atmospheric_modeling/data_parameter"
DST_DIR="./atmospheric_modeling/data_parameter_shuf"

TRAIN_DIR="$DST_DIR/train"
VALID_DIR="$DST_DIR/valid"
TEST_DIR="$DST_DIR/test"

mkdir -p "$TRAIN_DIR" "$VALID_DIR" "$TEST_DIR"

ALL_DIRS=$(find "$SRC_DIR"/{train,valid,test} -mindepth 1 -maxdepth 1 -type d | shuf)
DIR_ARRAY=($ALL_DIRS)

TOTAL=${#DIR_ARRAY[@]}

N_TRAIN=120
N_VALID=40
N_TEST=40

for ((i=0; i<TOTAL; i++)); do
    SRC="${DIR_ARRAY[$i]}"
    if [ $i -lt $N_TRAIN ]; then
        mv "$SRC" "$TRAIN_DIR/"
    elif [ $i -lt $((N_TRAIN + N_VALID)) ]; then
        mv "$SRC" "$VALID_DIR/"
    elif [ $i -lt $((N_TRAIN + N_VALID + N_TEST)) ]; then
        mv "$SRC" "$TEST_DIR/"
    else
        echo "Warning: Extra directory ${SRC} not moved (exceeds N_TRAIN + N_VALID + N_TEST)"
    fi
done
