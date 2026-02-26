#!/bin/bash
set -e  

# LATENT_TAG="ckpt_1699"
# NUM_LATENT=10
# BASE_PATH="saved_model/latent_10/$LATENT_TAG"

# MODEL_PATH="saved_model/latent_10/ckpt_1699.pt"
# OBS_PATH="$BASE_PATH/obs_data.pth"
# LSTM_PATH="$BASE_PATH/lstm.pth"
# SAVE_PATH="$BASE_PATH"
# WANDB_NAME="$LATENT_TAG"
# DEVICE="cuda:0"

NUM_LATENT=10
BASE_PATH="saved_model/grid_latent_10"

DATA_PATH="data/data_2000_150x150_coriolis_normalized_ic_grid.npz"
MODEL_PATH="$BASE_PATH/ckpt_1999.pt"
OBS_PATH="$BASE_PATH/obs_data.pth"
LSTM_PATH="$BASE_PATH/lstm.pth"
SAVE_PATH="$BASE_PATH"
WANDB_NAME="$LATENT_TAG"
DEVICE="cuda:1"

mkdir -p "$BASE_PATH"

cd /work/pengpeng/data-assimilation/shallow_water

# echo "Running data_observation_coriolis_ic.py..."
# python data_observation_coriolis_ic_ckpt.py \
#     --data-path "$DATA_PATH" \
#     --model-path "$MODEL_PATH" \
#     --obs-path "$OBS_PATH" \
#     --num-latent-states "$NUM_LATENT" \
#     --device "$DEVICE"

# echo "Running train_lstm_ic.py..."
# python train_lstm_ic_ckpt.py \
#     --data-path "$OBS_PATH" \
#     --save-path "$LSTM_PATH" \
#     --wandb-name "$WANDB_NAME" \
#     --num-latent-states "$NUM_LATENT" \
#     --device "$DEVICE"

echo "Running test_batch_coriolis_ic_ensemble_ckpt.py..."
python test_batch_coriolis_ic_ensemble_ckpt.py \
    --model-path "$MODEL_PATH" \
    --lstm-path "$LSTM_PATH" \
    --save-path "$SAVE_PATH" \
    --data-path "$OBS_PATH" \
    --noise-level 0.1 \
    --num-latent-states "$NUM_LATENT" \
    --device "$DEVICE"

echo "All scripts completed."
