#!/bin/bash

SRC_DIR="/work/pengpeng/data-assimilation/planetswe/data_parameter"
DST_DIR="/work/pengpeng/data-assimilation/planetswe/data_parameter_shuf"

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
