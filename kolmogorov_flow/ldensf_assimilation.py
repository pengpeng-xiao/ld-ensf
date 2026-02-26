import numpy as np
import torch
import torch.nn as nn
import wandb
import deepxde as dde

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src import utils
from src.data_preprocess import DataPreprocessor, select_space_subset
from src.model import ResFourierLDNN
from src.normalization import Normalize

def create_training_options():
    parser = argparse.ArgumentParser()
    
    # --------------- path and logging ---------------
    parser.add_argument("--base-path",          type=Path,  default=None, help="base path for data and models")
    parser.add_argument("--data-path",          type=Path,  default="data/data_observation_500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14.pth")
    parser.add_argument("--model-path",         type=Path,  default="saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14")
    
    # --------------- model parameter ---------------
    parser.add_argument("--num-latent-states",  type=int,   default=9)
    parser.add_argument("--fourier-mapping-size", type=int, default=10)
    parser.add_argument("--NN-dyn-depth",       type=int,   default=9)
    parser.add_argument("--NN-dyn-width",       type=int,   default=200)
    parser.add_argument("--NN-rec-depth",       type=int,   default=14)
    parser.add_argument("--NN-rec-width",       type=int,   default=500)
    parser.add_argument("--activation",         type=str,   default="relu")
    parser.add_argument("--kernel-initializer", type=str,   default="Glorot normal")
    parser.add_argument("--const-u",            action="store_true",  default=False)
    
    # --------------- training parameter ---------------
    parser.add_argument("--device",             type=str,   default="cuda:3")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--batch-size",         type=int,   default=2)
    parser.add_argument("--learning-rate",      type=float, default=1e-3)
    parser.add_argument("--num-epochs",         type=int,   default=2000)
    parser.add_argument("--eval-interval",      type=int,   default=20)
    parser.add_argument("--lr-gamma",           type=float, default=0.6,    help="learning rate decay factor")

    # --------------- assimilation parameter ---------------
    parser.add_argument("--eps-alpha",          type=float, default=0.01)
    parser.add_argument("--euler-steps",        type=int,   default=100)
    parser.add_argument("--scaling",            type=float, default=100)
    parser.add_argument("--obs-sigma",          type=float, default=0.1)
    parser.add_argument("--noise-level",        type=float, default=0.1)
    parser.add_argument("--ensemble-size",      type=int,   default=20)
    opt = parser.parse_args()
    return opt
    
def main(opt):
    ensemble_size = opt.ensemble_size
    noise_level = opt.noise_level
    obs_sigma = opt.obs_sigma
    scaling = opt.scaling
    
    eps_alpha = opt.eps_alpha
    euler_steps = opt.euler_steps
    utils.set_seed(opt.seed)  

    wandb.init(entity="20307110428", project="KF_DA", name=f"avg_n_{noise_level}_o_{obs_sigma}_s_{scaling}_rmse",
        dir=str(opt.base_path / f"saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14"), config=vars(opt), save_code=True)

    dataset = torch.load(opt.base_path / opt.data_path, map_location=opt.device, weights_only=True)
    data_test = dataset["data_test"]
    data_train = dataset["data_train"]
    data_valid = dataset["data_valid"]

    for data in [data_test]:
        data["observations"] = data["observation"][:ensemble_size]
        data["y"] = data["y"][:ensemble_size]
        data["x"] = data["x"][:ensemble_size]
        data["u"] = data["u"][:ensemble_size]
        data["latent_states"] = data["latent_states"][:ensemble_size]
        # move data to device
        for key in ["u", "x", "observations", "dt"]:
            data[key] = data[key].to(opt.device)

    dim_u = data_test["u"].shape[-1]
    dim_x = data_test["x"].shape[-1]
    dim_y = data_test["y"].shape[-1]
    
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

    encoder = torch.load(opt.base_path / "saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14/lstm.ckpt", map_location=opt.device)
    encoder.eval()
    model.to(opt.device)
    model = utils.load_model(model, opt.base_path / "saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14/dyn_1999.ckpt",
                       opt.base_path / "saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14/retrained_rec_1999.ckpt", opt.device)

    from src.normalization import Normalize_gaussian, Normalize
    stat = np.load(opt.base_path / "saved_model/cplx_Re500_1500_150x150/mean_std.npz", allow_pickle=True)
    mean = stat["mean"]
    std = stat["std"]
    normalize_y = Normalize_gaussian(mean, std)
    normalize_u = Normalize([500], [1500])
        
    rmses_state, rmses_latent, rmses_u = [], [], []
    rmses_state_orig, rmses_latent_orig, rmses_u_orig = [], [], []
    for true_traj in range(10):
        observation = data_test["observation"][true_traj]
        true_state = data_test["y"][true_traj].detach().cpu().numpy()
        true_latent = data_test["latent_states"][true_traj].detach().cpu().numpy()
        true_u = data_test["u"][true_traj].detach().cpu().numpy()
        true_u = np.repeat(true_u[None,:], true_state.shape[0], axis=0)
        orig_u = np.repeat(data_test["u"].detach().cpu().numpy()[:,None,:], true_state.shape[0], axis=1)
        
        # optimal_latent = torch.load(opt.base_path / "saved_model/150x150_non_const/optimal_test_latent.pth", map_location=opt.device)[:20]
        assimilated_states = model.forward_data_assimilation_ensf(data=data_test, observations=observation, encoder=encoder, noise_level=noise_level, 
                                                                device=opt.device, optimal_latent=None, scaling=scaling, obs_sigma=obs_sigma, 
                                                                eps_alpha=eps_alpha, euler_steps=euler_steps).detach().cpu().numpy()
        
        # denormalize
        assimilated_states = normalize_y.normalize_back(assimilated_states)
        true_state = normalize_y.normalize_back(true_state)
        
        rmse_state = np.sqrt(np.mean((assimilated_states - true_state[np.newaxis,:,:,:])**2, axis=(2,3)) / np.mean(true_state[np.newaxis,:,:,:]**2, axis=(2,3)))
        rmses_state.append(rmse_state)
        
        assimilated_latent, assimilated_u, encoded_latent = model.forward_data_assimilation_ensf(data=data_test, observations=observation, encoder=encoder, noise_level=noise_level, 
                                                                              device=opt.device, latent_state = True, optimal_latent=None, 
                                                                              scaling = scaling, obs_sigma =  obs_sigma, eps_alpha = eps_alpha, euler_steps =euler_steps)
        assimilated_latent = assimilated_latent.detach().cpu().numpy()
        encoded_latent = encoded_latent.detach().cpu().numpy()
        assimilated_u = assimilated_u.detach().cpu().numpy()
        
        rmse_latent = np.sqrt(np.mean((assimilated_latent - true_latent[np.newaxis,:,:])**2, axis=(2)) / np.mean(true_latent[np.newaxis,:,:]**2, axis=(2)))
        rmses_latent.append(rmse_latent)
        
        # denormalize u
        assimilated_u = normalize_u.normalize_back(assimilated_u)
        true_u = normalize_u.normalize_back(true_u)
        
        rmse_u = np.sqrt(np.mean((assimilated_u - true_u[np.newaxis,:,:])**2, axis=(2)) / np.mean(true_u[np.newaxis,:,:]**2, axis=(2)))
        rmses_u.append(rmse_u)
        
        with torch.no_grad():
            model.eval()
            orig = model(data_test, opt.device).detach().cpu().numpy()
            orig_latent = model(data_test, opt.device, latent_state = True).detach().cpu().numpy()
        
        # denormalize origin data
        orig = normalize_y.normalize_back(orig)
        orig_u = normalize_u.normalize_back(orig_u)
        
        rmse_state_orig = np.sqrt(np.mean((orig - true_state[np.newaxis,:,:,:])**2, axis=(2,3)) / np.mean(true_state[np.newaxis,:,:,:]**2, axis=(2,3)))
        rmses_state_orig.append(rmse_state_orig)
        rmse_latent_orig = np.sqrt(np.mean((orig_latent - true_latent[np.newaxis,:,:])**2, axis=(2)) / np.mean(true_latent[np.newaxis,:,:]**2, axis=(2)))
        rmses_latent_orig.append(rmse_latent_orig)
        rmse_u = np.sqrt(np.mean((orig_u - true_u[np.newaxis,:,:])**2, axis=(2)) / np.mean(true_u[np.newaxis,:,:]**2, axis=(2)))
        rmses_u_orig.append(rmse_u)
        
        print(f"finished {true_traj}th trajectory")
    
    np.savez_compressed(opt.base_path / "saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_gamma_0.5_resd_14" /f"ensemble_assimilation_output_avg_n_{noise_level}_o_{obs_sigma}_s_{scaling}_seed_{opt.seed}_rmse.npz",
                        rmses_state = rmses_state, rmses_state_orig = rmses_state_orig,
                        rmses_latent = rmses_latent, rmses_latent_orig = rmses_latent_orig,
                        rmses_u = rmses_u, rmses_u_orig = rmses_u_orig)
    
    wandb.log(
        {"state": wandb.plot.line_series(xs=np.arange(40)*5, 
            ys=[np.mean(rmses_state, axis=0), np.mean(rmses_state_orig, axis=0)], keys=["rmse state", "rmse state orig"], 
            title="State RMSE", xname="time steps", ),
        "latent": wandb.plot.line_series(xs=np.arange(40)*5,
            ys=[np.mean(rmses_latent, axis=0), np.mean(rmses_latent_orig, axis=0)], keys=["rmse latent", "rmse latent orig"],
            title="Latent RMSE", xname="time steps", ),
        "u": wandb.plot.line_series(xs=np.arange(40)*5,
            ys=[np.mean(rmses_u, axis=0), np.mean(rmses_u_orig, axis=0)], keys=["rmse u", "rmse u orig"],
            title="U RMSE", xname="time steps", )})


if __name__ == "__main__":
    dde.config.real.set_float32()
    torch.set_default_device("cpu") # to make sure dataloader is on cpu
    opt = create_training_options()
    main(opt)
        