import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import wandb
import deepxde as dde
import h5py

import sys
import argparse
from pathlib import Path
import shutil
import glob
from tqdm import tqdm
import re

from src import utils
from src.logger import Logger
from src.data_preprocess import DataPreprocessor, select_space_subset
from src.model import LDNN
from src.train import Trainer
from src.normalization import Normalize, Normalize_gaussian

def create_training_options():
    parser = argparse.ArgumentParser()
    
    # --------------- path and logging ---------------
    parser.add_argument("--base-path",          type=Path,  default=None, help="base path for data and models")
    parser.add_argument("--data-path",          type=Path,  default="data_parameter_shuf")
    parser.add_argument("--log-dir",            type=Path,  default="log")
    parser.add_argument("--wandb-entity",       type=str,   default="20307110428",    help="user name of your W&B account")
    parser.add_argument("--wandb-project",      type=str,   default="planetswe",           help="name of the W&B project")
    parser.add_argument("--name",               type=str,   default="parameter_shuf_t21d_orig", help="name of the run")
    parser.add_argument("--model-path",         type=Path,  default="saved_model/parameter_shuf_t21d_orig")
    
    # --------------- dataset parameter ---------------
    parser.add_argument("--num-points-train",   type=int,   default=5000,   help="number of spatial points in training set")
    parser.add_argument("--num-points-valid",   type=int,   default=5000,   help="number of spatial points in validation set")
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

def parse_parameters_from_name(name):
    pattern = r"hf0_([0-9p]+)_sigma_([0-9p]+)"
    match = re.search(pattern, name)
    if match:
        hf0_str = match.group(1).replace("p", ".")
        sigma_str = match.group(2).replace("p", ".")
        hf0 = float(hf0_str)
        sigma = float(sigma_str)
        return hf0, sigma
    else:
        raise ValueError(f"Could not parse parameters from name: {name}")

def load_from_hdf5(file_list):
    dataset = {}
    h, u, coords, hf0s, sigmas = [], [], [], [], []
    for fname in tqdm(file_list):
        with h5py.File(fname, "r") as f: 
            path = Path(fname)
            foldername = path.name
            hf0, sigma = parse_parameters_from_name(foldername)         
            hf0s.append(hf0)
            sigmas.append(sigma)  
            h.append(f["tasks/h"][::2][:63])
            u.append(f["tasks/u"][::2][:63])
            phi = f["scales/phi_hash_7b8ec7cabc40ac4b596a5ef833e9eab019f07d46"][:]
            theta = f["scales/theta_hash_7371cda98b66ed3211b41cdb54c08495aa28ea62"][:]
            phi_grid, theta_grid = np.meshgrid(phi, theta)
            # phi_grid = phi_grid.T
            # theta_grid = theta_grid.T
            coords.append(np.stack([phi_grid.ravel(), theta_grid.ravel()], axis=-1))
    h = np.stack(h, axis=0).astype(np.float32)
    print("==========h.shape==========", h.shape)
    u = np.stack(u, axis=0).transpose(0,1,3,4,2).astype(np.float32)
    print("==========u.shape==========", u.shape)
    dataset["y"] = np.concatenate([h[...,np.newaxis], u], axis=-1).astype(np.float32)
    print(dataset["y"].shape)
    # dataset["u"] = dataset["y"][:,1].astype(np.float32)
    dataset["x"] = np.stack(coords, axis=0).astype(np.float32)
    print("dataset['x'].shape", dataset["x"].shape)
    hf0s = np.stack(hf0s, axis=0)
    sigmas = np.stack(sigmas, axis=0)
    dataset["u"] = np.stack((hf0s, sigmas), axis=-1).astype(np.float32)
    print("u shape", dataset["u"].shape)
    # dataset["x"] = coords
    return dataset
    
def load_data(opt, log):
    valid_file_list = sorted(glob.glob(str(opt.base_path / opt.data_path / "valid/IC_*/IC_*_s1.h5")))
    data_valid = load_from_hdf5(valid_file_list)
    train_file_list = sorted(glob.glob(str(opt.base_path / opt.data_path / "train/IC_*/IC_*_s1.h5")))
    data_train = load_from_hdf5(train_file_list)
    test_file_list = sorted(glob.glob(str(opt.base_path / opt.data_path / "test/IC_*/IC_*_s1.h5")))
    data_test = load_from_hdf5(test_file_list)
    
    
    for dataset in [data_train, data_valid, data_test]:
        # --------------- select needed time slices and reshape ---------------
        interval = opt.interval 
        dataset["y"] = dataset["y"][:, ::1, :, :, :] #! we don't deal with time slice in this version
        B, T, H, W, C = dataset["y"].shape
        # new_shape = dataset['y'].shape[:-2] + (-1,)
        dataset["y"] = dataset["y"].reshape(B, T, H * W, C)
        # dataset["u"] = np.tile(dataset["u"][:, None, :], (1, dataset["y"].shape[1], 1))
        # dataset["x"] = np.broadcast_to(dataset["x"][None,None,:,:], (dataset["y"].shape[0], 
        #             dataset["y"].shape[1], dataset["x"].shape[-2], dataset["x"].shape[-1]))
        dataset["x"] = np.broadcast_to(dataset["x"][:,None,:,:], (dataset["y"].shape[0], 
                dataset["y"].shape[1], dataset["x"].shape[-2], dataset["x"].shape[-1]))
        # ------------------ normalize dataset ------------------
        # dt_normalize = opt.dt_normalize / interval
        dataset["dt"] = np.array(opt.dt).astype(np.float32)
        normlize_x = Normalize([0, 0], [2 * np.pi, np.pi])
        dataset["x"] = normlize_x.normalize_forw(dataset["x"])
        if dataset is data_train:
            mean = np.mean(data_train["y"], axis=tuple(range(data_train["y"].ndim-1)))
            std = np.std(data_train["y"], axis=tuple(range(data_train["y"].ndim-1)))
            np.savez(opt.base_path / opt.model_path / "mean_std.npz", mean=mean, std=std)
            normalize_y = Normalize_gaussian(mean, std)
        dataset["y"] = normalize_y.normalize_forw(dataset["y"])
        
        normalize_u = Normalize([0.1, 1.0], [30.0, 4.0])
        dataset["u"] = normalize_u.normalize_forw(dataset["u"])
    
    data_train = select_space_subset(data_train, opt.num_points_train)
    data_valid = select_space_subset(data_valid, opt.num_points_valid)
    
    np.savez_compressed(opt.base_path / opt.data_path / "data_preprocessed_n5000_t21d.npz", data_train=data_train, data_valid=data_valid, data_test=data_test)
    
    return data_train, data_valid, data_test
    
def main(opt):
    log = Logger(log_dir=opt.base_path /opt.model_path / opt.log_dir)
    log.info("=======================================================")
    log.info("                  Data Assimilation                     ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")
    
    wandb.init(entity=opt.wandb_entity, project=opt.wandb_project, name=opt.name, 
               dir=str(opt.base_path / opt.model_path / opt.log_dir), config=vars(opt), save_code=True)
    
    utils.set_seed(opt.seed)  

    data_train, data_valid, data_test = load_data(opt, log)
    # stop here
    return data_train, data_valid, data_test
    data = np.load(opt.base_path / opt.data_path /"data_preprocessed_n5000_t21d.npz", allow_pickle = True)
    data_train = data["data_train"].item()
    data_valid = data["data_valid"].item()
    data_test = data["data_test"].item()
    log.info("Data loaded.")
    
    # data_train = select_space_subset(data_train, opt.num_points_train)
    # data_valid = select_space_subset(data_valid, opt.num_points_valid)

    dim_x = data_train["x"].shape[-1]
    dim_y = data_train["y"].shape[-1]
    
    # Define model
    input_shape_r = opt.num_latent_states + dim_x
    model = LDNN(
                # opt.fourier_mapping_size,
                [opt.num_latent_states + 2] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states],
                [input_shape_r] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y],
                activation=opt.activation,
                kernel_initializer=opt.kernel_initializer,
    )
    # model = torch.nn.DataParallel(model)
    model.to(opt.device)
    # model.cuda()
    criterion = nn.MSELoss()
    # from src.loss import RelativeL2Loss
    # criterion = RelativeL2Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=opt.lr_gamma)

    trainer = Trainer(model, optimizer, criterion, \
                      batch_size=opt.batch_size, device=opt.device, lr_scheduler=scheduler, )

    # Train model
    save_path = opt.base_path / opt.model_path 
    trainer.train(data_train, data_valid, num_epochs=opt.num_epochs, eval_interval=opt.eval_interval, save_path=save_path)
    
    log.info("Training completed.")
    test_error = trainer.test(data_test)
    wandb.log({"test_error": test_error})
    
    # torch.save(model.dyn.state_dict(), opt.base_path / opt.model_path / "dyn.pth")
    # torch.save(model.rec.state_dict(), opt.base_path/ opt.model_path / "rec.pth")
    # print(f"Model saved to {opt.model_path}")
    wandb.save(opt.model_path, base_path=opt.base_path)
    
    log.info("Model saved. Finish training.")
    
    current_file = __file__
    destination_file = opt.base_path / opt.model_path / "main.py"
    shutil.copyfile(current_file, destination_file)
    
    log.info("Code saved.")
    wandb.finish()

if __name__ == "__main__":
    dde.config.real.set_float32()
    torch.set_default_device("cpu")
    opt = create_training_options()
    main(opt)