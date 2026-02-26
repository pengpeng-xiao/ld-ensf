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
from src.model import ResFourierLDNN
from src.normalization import Normalize, Normalize_gaussian

def create_training_options():
    parser = argparse.ArgumentParser()
    
    # --------------- path and logging ---------------
    parser.add_argument("--base-path",          type=Path,  default=None, help="base path for data and models")
    parser.add_argument("--data-path",          type=Path,  default="data/data_cplx_Re_500_1500_150x150.npz")
    parser.add_argument("--log-dir",            type=Path,  default="log")
    parser.add_argument("--name",               type=str,   default="run_data_assimilation",    help="name of the run")
    parser.add_argument("--model-path",         type=Path,  default="saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14")
    
    # --------------- dataset parameter ---------------
    parser.add_argument("--prop-train",         type=float, default=0.6)
    parser.add_argument("--prop-valid",         type=float, default=0.2)  
    parser.add_argument("--prop-test",          type=float, default=0.2)
    # parser.add_argument("--prop-train",         type=float, default=0.0)
    # parser.add_argument("--prop-valid",         type=float, default=0.0)  
    # parser.add_argument("--prop-test",          type=float, default=1.0)
    parser.add_argument("--interval",           type=int,   default=5,     help="time interval between two time slices")
    parser.add_argument("--dt-normalize",       type=float, default=1.0)
    
    # --------------- model parameter ---------------
    parser.add_argument("--num-latent-states",  type=int,   default=9)
    parser.add_argument("--fourier-mapping-size", type=int, default=10)
    parser.add_argument("--NN-dyn-depth",       type=int,   default=9)
    parser.add_argument("--NN-dyn-width",       type=int,   default=200)
    parser.add_argument("--NN-rec-depth",       type=int,   default=14)
    parser.add_argument("--NN-rec-width",       type=int,   default=500)
    parser.add_argument("--activation",         type=str,   default="relu")
    parser.add_argument("--kernel-initializer", type=str,   default="Glorot normal")
    
    # --------------- training parameter ---------------
    parser.add_argument("--device",             type=str,   default="cuda:3")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--batch-size",         type=int,   default=2)

    opt = parser.parse_args()
    return opt

def load_data(opt):
    dataset = np.load(Path(opt.base_path) / Path(opt.data_path), allow_pickle = True)
    dataset = dict(dataset)
    dataset = {key: dataset[key].astype(np.float32) for key in dataset.keys()}
    
    # --------------- select needed time slices and reshape ---------------
    interval = opt.interval 
    dataset["y"] = dataset["y"][:, ::interval, :, :, :]
    new_shape = dataset['y'].shape[:-2] + (-1,)
    dataset["y"] = dataset["y"].reshape(new_shape).transpose(0,1,3,2)
    
    dataset["u"] = dataset["u"][:, None]
    # dataset["u"] = np.tile(dataset["u"][:, None, None], (1, dataset["y"].shape[1], 1))
    
    dataset["x"] = np.broadcast_to(dataset["x"][None,None,:,:], (dataset["y"].shape[0], 
                dataset["y"].shape[1], dataset["x"].shape[-2], dataset["x"].shape[-1]))
    
    # ------------------ normalize dataset ------------------
    dt_normalize = opt.dt_normalize / interval
    dataset["dt"] = np.array(dataset["dt"] / dt_normalize, dtype=np.float32)

    normlize_x = Normalize([0, 0], [2 * np.pi, 2 * np.pi])
    normalize_u = Normalize([500], [1500])
    dataset["x"] = normlize_x.normalize_forw(dataset["x"])
    dataset["u"] = normalize_u.normalize_forw(dataset["u"])
    
    # ------ split dataset into train, valid and test ------
    prep_data = DataPreprocessor(dataset, prop_train=opt.prop_train, prop_valid=opt.prop_valid, prop_test=opt.prop_test)
    data_train = prep_data.get_train_data()
    data_valid = prep_data.get_valid_data()
    data_test = prep_data.get_test_data()
    
    from src.normalization import Normalize_gaussian
    stat = np.load(opt.base_path / "saved_model/cplx_Re500_1500_150x150/mean_std.npz", allow_pickle=True)
    mean = stat["mean"]
    std = stat["std"]
    normalize_y = Normalize_gaussian(mean, std)
    data_train["y"] = normalize_y.normalize_forw(data_train["y"])
    data_valid["y"] = normalize_y.normalize_forw(data_valid["y"])
    data_test["y"] = normalize_y.normalize_forw(data_test["y"])
    
    return data_train, data_valid, data_test

def load_model(model, path_dyn, path_rec, device):
    model.dyn.load_state_dict(torch.load(path_dyn, map_location=device, weights_only=True))
    model.rec.load_state_dict(torch.load(path_rec, map_location=device, weights_only=True))
    return model

def observ_idx(Nx, Ny, interval):
    grid_inx_x, grid_inx_y = np.meshgrid(np.arange(0, Nx, interval), np.arange(0, Ny, interval))
    flat_idx = grid_inx_y + grid_inx_x * Ny
    return flat_idx.ravel()
    
def main(opt):
    
    utils.set_seed(opt.seed)  

    data_train, data_valid, data_test = load_data(opt)

    flat_idx = observ_idx(150, 150, 15)
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
    # model = load_model(model, opt.base_path / opt.model_path / "dyn_1999.ckpt",
    #                    opt.base_path / opt.model_path / "rec_1999.ckpt", opt.device)
    torch.load(opt.base_path / "saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_freeze_gamma_0.5_resd_14/ckpt_1999.pt", map_location=opt.device)
    
    
    for data in [data_train, data_valid, data_test]:
        with torch.no_grad():
            latent_states = model(data, opt.device, latent_state=True)
            data["latent_states"] = latent_states

    data = {
        "data_train": data_train,
        "data_valid": data_valid,
        "data_test": data_test,
    }
    
    torch.save(data, opt.base_path / "data/data_observation_500_1500_150x150_resnet_fourier_10_freeze_gamma_0.5_resd_14.pth")
    

if __name__ == "__main__":
    dde.config.real.set_float32()
    torch.set_default_device("cpu") # to make sure dataloader is on cpu
    opt = create_training_options()
    main(opt)