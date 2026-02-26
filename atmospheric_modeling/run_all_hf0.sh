#!/bin/bash

# IC_INDEX=1
# IC_DIR="./data_stubs/"
# OUTPUT_DIR="./data_parameter/"
# H_F0_LIST=(0.5 20.0)
# SIGMA=2 * np.pi / 3

# trap "echo 'Interrupted. Exiting loop.'; exit 1" SIGINT

# for HF0 in "${H_F0_LIST[@]}"
# do
#     echo "Running with h_f0=${HF0}, saving to ${OUTPUT_DIR}"
    
#     mpiexec -n 64 python gen_SWE_force.py \
#         --index $IC_INDEX \
#         --ic_dir $IC_DIR \
#         --output_dir $OUTPUT_DIR \
#         --hf0 $HF0 \
#         --sigma $SIGMA
# done

IC_INDEX=1
IC_DIR="./data_stubs/"
OUTPUT_DIR="./data_parameter/"
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

