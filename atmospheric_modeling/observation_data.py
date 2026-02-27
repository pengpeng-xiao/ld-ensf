import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import wandb
import deepxde as dde
import h5py

import argparse
from pathlib import Path
import shutil
import glob
from tqdm import tqdm
import re

from src import utils
from src.logger import Logger
from src.data_preprocess import DataPreprocessor, select_space_subset
from src.model import ResFourierLDNN
from src.train import Trainer
from src.normalization import Normalize, Normalize_gaussian

def create_training_options():
    parser = argparse.ArgumentParser()
    
    # --------------- path and logging ---------------
    parser.add_argument("--base-path",          type=Path,  default=None, help="base path for data and models")
    parser.add_argument("--data-path",          type=Path,  default="atmospheric_modeling/data_parameter_shuf")
    parser.add_argument("--log-dir",            type=Path,  default="log")
    parser.add_argument("--name",               type=str,   default="ldnet",    help="name of the run")
    parser.add_argument("--model-path",         type=Path,  default="atmospheric_modeling/saved_model/ldnet")
    
    # --------------- dataset parameter ---------------
    parser.add_argument("--num-points-train",   type=int,   default=40000,   help="number of spatial points in training set")
    parser.add_argument("--num-points-valid",   type=int,   default=40000,   help="number of spatial points in validation set")
    parser.add_argument("--prop-train",         type=float, default=0.6)
    parser.add_argument("--prop-valid",         type=float, default=0.2)  
    parser.add_argument("--prop-test",          type=float, default=0.2) 
    parser.add_argument("--interval",           type=int,   default=20,     help="time interval between two time slices")
    parser.add_argument("--dt",       type=float, default=0.04)
    
    # --------------- model parameter ---------------
    parser.add_argument("--num-latent-states",  type=int,   default=50)
    parser.add_argument("--fourier-mapping-size",type=int,   default=50)
    parser.add_argument("--NN-dyn-depth",       type=int,   default=8)
    parser.add_argument("--NN-dyn-width",       type=int,   default=200)
    parser.add_argument("--NN-rec-depth",       type=int,   default=15)
    parser.add_argument("--NN-rec-width",       type=int,   default=500)
    parser.add_argument("--activation",         type=str,   default="relu")
    parser.add_argument("--kernel-initializer", type=str,   default="Glorot normal")
    parser.add_argument("--dropout",            type=float, default=0)
    parser.add_argument("--const-u",            action="store_true",  default=False) 
    
    # --------------- training parameter ---------------
    parser.add_argument("--device",             type=str,   default="cuda:0")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--batch-size",         type=int,   default=2)
    parser.add_argument("--learning-rate",      type=float, default=1e-3)
    parser.add_argument("--num-epochs",         type=int,   default=2000)
    parser.add_argument("--eval-interval",      type=int,   default=20)
    parser.add_argument("--lr-gamma",           type=float, default=0.6,    help="learning rate decay factor")

    opt = parser.parse_args()
    return opt

def load_model(model, path_dyn, path_rec, device):
    model.dyn.load_state_dict(torch.load(path_dyn, map_location=device, weights_only=True))
    model.rec.load_state_dict(torch.load(path_rec, map_location=device, weights_only=True))
    return model

def observ_idx(Nx, Ny, interval):
    grid_inx_x, grid_inx_y = np.meshgrid(np.arange(0, Nx, interval), np.arange(0, Ny, interval))
    flat_idx = grid_inx_y + grid_inx_x * Ny
    return flat_idx.ravel()

def extract_essential(data_dict):
    return {
        "latent_states": data_dict["latent_states"],
        "observation": data_dict["observation"],
        "u": data_dict["u"],
    }
    
def main(opt):
    
    utils.set_seed(opt.seed)  

    data = np.load(opt.base_path / opt.data_path /"data_normalized.npz", allow_pickle = True)
    data_train = data["data_train"].item()
    data_valid = data["data_valid"].item()
    data_test = data["data_test"].item()

    flat_idx = observ_idx(512, 256, 16)
    # flat_idx = np.random.choice(150*150, 100, replace=False)
    for data in [data_train, data_valid, data_test]:
        data["observation"] = data["y"][:, :, flat_idx, :]
        # move data to device
        for key in data.keys():
            data[key] = torch.from_numpy(data[key]).to(opt.device)

    dim_u = data_train["u"].shape[-1]
    dim_x = data_train["x"].shape[-1]
    dim_y = data_train["y"].shape[-1]
    
    # Define model
    input_shape_d = opt.num_latent_states + dim_u
    input_shape_r = opt.num_latent_states + dim_x
    model = ResFourierLDNN(
                opt.fourier_mapping_size,
                [input_shape_d] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states],
                [input_shape_r] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y],
                activation=opt.activation,
                kernel_initializer=opt.kernel_initializer,
    )

    model.to(opt.device)
    model = load_model(model, opt.base_path / opt.model_path / "dyn_1799.ckpt",
                       opt.base_path / opt.model_path / "rec_1799.ckpt", opt.device)
    
    for data in [data_train, data_valid, data_test]:
        with torch.no_grad():
            latent_states = model(data, opt.device, latent_state=True)
            data["latent_states"] = latent_states
    
    data = {
        "data_train": extract_essential(data_train),
        "data_valid": extract_essential(data_valid),
        "data_test": data_test,
    }
    
    torch.save(data, opt.base_path / opt.model_path /"observation_data.pth")
    

if __name__ == "__main__":
    dde.config.real.set_float32()
    torch.set_default_device("cpu") # to make sure dataloader is on cpu
    opt = create_training_options()
    main(opt)