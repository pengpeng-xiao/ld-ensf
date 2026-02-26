import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import wandb
import deepxde as dde

import argparse
from pathlib import Path

from src import utils
from src.data_preprocess import DataPreprocessor, select_space_subset
from src.resnet import ResLDNN
from src.normalization import Normalize

def create_training_options():
    parser = argparse.ArgumentParser()
    
    # --------------- path and logging ---------------
    parser.add_argument("--base-path",          type=Path,  default="/work/pengpeng/data-assimilation/shallow_water")
    parser.add_argument("--data-path",          type=Path,  default="data/data_2000_150x150_coriolis_ic.npz")
    parser.add_argument("--log-dir",            type=Path,  default="log")
    parser.add_argument("--wandb-entity",         type=str,   default="20307110428",            help="user name of your W&B account")
    parser.add_argument("--wandb-project",      type=str,   default="SW_vanilla_2000",          help="name of the W&B project")
    parser.add_argument("--name",               type=str,   default="coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u", help="name of the run")
    parser.add_argument("--model-path",         type=Path,  default="saved_model/coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u")
    
    # --------------- dataset parameter ---------------
    parser.add_argument("--Nt",                 type=int,   default=2000,   help="number of time slices in the original dataset")
    parser.add_argument("--nt",                 type=int,   default=400,    help="number of time slices in the reduced dataset")
    # parser.add_argument("--num-points-train",   type=int,   default=1000,   help="number of spatial points in training set")
    # parser.add_argument("--num-points-valid",   type=int,   default=1000,   help="number of spatial points in validation set")
    parser.add_argument("--prop-train",         type=float, default=0.0)
    parser.add_argument("--prop-valid",         type=float, default=0.0)  
    parser.add_argument("--prop-test",          type=float, default=1.0) 
    parser.add_argument("--interval",           type=int,   default=40,     help="time interval between two time slices")
    # parser.add_argument("--dt-normalize", type=float, default=500)
    
    # # --------------- model parameter ---------------
    parser.add_argument("--dt-normalize", type=float, default=580)
    
    # --------------- model parameter ---------------
    parser.add_argument("--num-latent-states",  type=int,   default=10)
    parser.add_argument("--NN-dyn-depth",       type=int,   default=8)
    parser.add_argument("--NN-dyn-width",       type=int,   default=50)
    parser.add_argument("--NN-rec-depth",       type=int,   default=10)
    parser.add_argument("--NN-rec-width",       type=int,   default=300)
    parser.add_argument("--activation",         type=str,   default="relu")
    parser.add_argument("--kernel-initializer", type=str,   default="Glorot normal")
    parser.add_argument("--const-u",            action="store_true",  default=False)
    
    # --------------- training parameter ---------------
    parser.add_argument("--device",             type=str,   default="cuda:1")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--batch-size",         type=int,   default=2)
    parser.add_argument("--learning-rate",      type=float, default=1e-3)
    parser.add_argument("--num-epochs",         type=int,   default=2000)
    parser.add_argument("--eval-interval",      type=int,   default=20)
    parser.add_argument("--lr-gamma",           type=float, default=0.6,    help="learning rate decay factor")

    opt = parser.parse_args()
    return opt

# def load_data(opt):
#     dataset = np.load(opt.base_path / opt.data_path, allow_pickle = True)
#     dataset = dict(dataset)
#     dataset = {key: dataset[key].astype(np.float32) for key in dataset.keys()}
    
#     # --------------- select needed time slices and reshape ---------------
#     interval = opt.interval // (opt.Nt // opt.nt)
#     dataset["y"] = dataset["y"][:, ::interval, :, :, :]
#     new_shape = dataset['y'].shape[:-2] + (-1,)
#     dataset["y"] = dataset["y"].reshape(new_shape).transpose(0,1,3,2)
#     # dataset["u"] = np.tile(dataset["u"][:, None, :], (1, dataset["y"].shape[1], 1))
#     dataset["x"] = np.broadcast_to(dataset["x"][None,None,:,:], (dataset["y"].shape[0], 
#                 dataset["y"].shape[1], dataset["x"].shape[-2], dataset["x"].shape[-1]))
    
#     # ------------------ normalize dataset ------------------
#     dt_normalize = opt.dt_normalize / interval
#     dataset["dt"] = np.array(dataset["dt"] / dt_normalize, dtype=np.float32)
#     print("dt", dataset["dt"])
#     L_X1 = 1E+6
#     L_X2 = 1E+6
#     normlize_x = Normalize([- L_X1 / 2, - L_X2 / 2], [L_X1 / 2, L_X2 / 2])
#     normalize_y = Normalize([-0.2, -0.2, -0.5], [0.2, 0.2, 1.0])
#     dataset["x"] = normlize_x.normalize_forw(dataset["x"])
#     dataset["y"] = normalize_y.normalize_forw(dataset["y"])
    
#     # ------ split dataset into train, valid and test ------
#     # prep_data = DataPreprocessor(dataset, prop_train=opt.prop_train, prop_valid=opt.prop_valid, prop_test=opt.prop_test, shuffle=True)
#     prep_data = DataPreprocessor(dataset, prop_train=opt.prop_train, prop_valid=opt.prop_valid, prop_test=opt.prop_test, shuffle=False)
#     data_train = prep_data.get_train_data()
#     data_valid = prep_data.get_valid_data()
#     data_test = prep_data.get_test_data()
    
#     return data_train, data_valid, data_test

def load_model(model, path_dyn, path_rec, device):
    # if path_ic:
    #     model.ic.load_state_dict(torch.load(path_ic, map_location=device))
    model.dyn.load_state_dict(torch.load(path_dyn, map_location=device))
    model.rec.load_state_dict(torch.load(path_rec, map_location=device))
    return model

def observ_idx(Nx, Ny, interval):
    grid_inx_x, grid_inx_y = np.meshgrid(np.arange(0, Nx, interval), np.arange(0, Ny, interval))
    flat_idx = grid_inx_y + grid_inx_x * Ny
    return flat_idx.ravel()
    
def main(opt):
    
    utils.set_seed(opt.seed)  

    data = np.load(opt.base_path / "data/data_2000_150x150_coriolis_normalized_ic_nor_u.npz", allow_pickle = True)
    data_train = data["data_train"].item()
    data_valid = data["data_valid"].item()
    data_test = data["data_test"].item()

    flat_idx = observ_idx(150, 150, 15)
    # flat_idx = np.random.choice(150*150, 100, replace=False)# flat_idx = np.random.choice(150*150, 100, replace=False)
    
    for data in [data_train, data_valid, data_test]:
        # data["x"] = data["x"][:, :, flat_idx, :]
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
    model = ResLDNN(
                [input_shape_d] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states],
                [input_shape_r] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y],
                activation=opt.activation,
                kernel_initializer=opt.kernel_initializer,
                # dropout=opt.dropout
    )

    model.to(opt.device)
    model = load_model(model, 
                       opt.base_path / opt.model_path / "dyn_1999.ckpt",
                       opt.base_path / opt.model_path / "rec_1999.ckpt", opt.device)
    
    for data in [data_train, data_valid, data_test]:
        with torch.no_grad():
            latent_states = model(data, opt.device, latent_state=True)
            data["latent_states"] = latent_states

    data = {
        "data_train": data_train,
        "data_valid": data_valid,
        "data_test": data_test,
    }
    
    torch.save(data, opt.base_path / "data/data_2000_150x150_coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u.pth")
    

if __name__ == "__main__":
    dde.config.real.set_float32()
    torch.set_default_device("cpu") # to make sure dataloader is on cpu
    opt = create_training_options()
    main(opt)