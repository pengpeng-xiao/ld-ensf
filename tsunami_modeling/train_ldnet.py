import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import wandb
import deepxde as dde

import sys
import argparse
from pathlib import Path
import shutil

from src import utils
from src.logger import Logger
from src.data_preprocess import DataPreprocessor, select_space_subset
from src.model import ResLDNN
from src.train import Trainer
from src.normalization import Normalize, Normalize_gaussian

def create_training_options():
    parser = argparse.ArgumentParser()
    
    # --------------- path and logging ---------------
    parser.add_argument("--base-path",          type=Path,  default=None, help="base path for data and models")
    parser.add_argument("--data-path",          type=Path,  default="tsunami_modeling/data/tsunami_data.npz")
    parser.add_argument("--log-dir",            type=Path,  default="log")
    parser.add_argument("--wandb-entity",       type=str,   default=None,    help="user name of your W&B account")
    parser.add_argument("--wandb-project",      type=str,   default=None,           help="name of the W&B project")
    parser.add_argument("--name",               type=str,   default="ldnet", help="name of the run")
    parser.add_argument("--model-path",         type=Path,  default="tsunami_modeling/saved_model/ldnet")
    
    # --------------- dataset parameter ---------------
    parser.add_argument("--Nt",                 type=int,   default=2000,    help="number of time slices in the original dataset")
    parser.add_argument("--nt",                 type=int,   default=50,     help="number of time slices in the reduced dataset")
    parser.add_argument("--num-points-train",   type=int,   default=2000,   help="number of spatial points in training set")
    parser.add_argument("--num-points-valid",   type=int,   default=2000,   help="number of spatial points in validation set")
    parser.add_argument("--prop-train",         type=float, default=0.6)
    parser.add_argument("--prop-valid",         type=float, default=0.2)  
    parser.add_argument("--prop-test",          type=float, default=0.2) 
    parser.add_argument("--interval",           type=int,   default=40,     help="time interval between two time slices")
    parser.add_argument("--dt-normalize",       type=float, default=580)
    
    # --------------- model parameter ---------------
    parser.add_argument("--num-latent-states",  type=int,   default=200)
    parser.add_argument("--NN-dyn-depth",       type=int,   default=8)
    parser.add_argument("--NN-dyn-width",       type=int,   default=50)
    parser.add_argument("--NN-rec-depth",       type=int,   default=10)
    parser.add_argument("--NN-rec-width",       type=int,   default=300)
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

def load_data(opt, log):
    dataset = np.load(opt.base_path / opt.data_path, allow_pickle = True)
    dataset = dict(dataset)
    dataset = {key: dataset[key].astype(np.float32) for key in dataset.keys()}
    
    # --------------- select needed time slices and reshape ---------------
    interval = opt.interval // (opt.Nt // opt.nt)
    print(dataset["y"].shape)
    dataset["y"] = dataset["y"][:, ::interval, :, :, :]
    print(dataset["y"].shape)
    new_shape = dataset['y'].shape[:-2] + (-1,)
    dataset["y"] = dataset["y"].reshape(new_shape).transpose(0,1,3,2)
    const = np.zeros((dataset["y"].shape[0], dataset["y"].shape[1]-1, dataset["u"].shape[-1]), dtype=np.float32)
    if opt.const_u:
        const = np.zeros((dataset["y"].shape[0], dataset["y"].shape[1]-1, dataset["u"].shape[-1]), dtype=np.float32)
        dataset["u"] = np.concatenate((dataset["u"][:, None, :], const), axis=1)
    else:
        dataset["u"] = np.tile(dataset["u"][:, None, :], (1, dataset["y"].shape[1], 1))
    
    dataset["x"] = np.broadcast_to(dataset["x"][None,None,:,:], (dataset["y"].shape[0], 
                dataset["y"].shape[1], dataset["x"].shape[-2], dataset["x"].shape[-1]))
    
    # ------------------ normalize dataset ------------------
    dt_normalize = opt.dt_normalize / interval
    dataset["dt"] = np.array(dataset["dt"] / dt_normalize, dtype=np.float32)
    L_X1 = 1E+6
    L_X2 = 1E+6
    normlize_x = Normalize([- L_X1 / 2, - L_X2 / 2], [L_X1 / 2, L_X2 / 2])
    dataset["x"] = normlize_x.normalize_forw(dataset["x"])
    
    normalize_y = Normalize([-0.2, -0.2, -0.5], [0.2, 0.2, 1.0])
    dataset["y"] = normalize_y.normalize_forw(dataset["y"])
    
    normalize_u = Normalize([0, 0], [0.5, 0.5])
    dataset["u"] = normalize_u.normalize_forw(dataset["u"])
    
    # ------ split dataset into train, valid and test ------
    prep_data = DataPreprocessor(dataset, prop_train=opt.prop_train, prop_valid=opt.prop_valid, prop_test=opt.prop_test)
    data_train = prep_data.get_train_data()
    data_valid = prep_data.get_valid_data()
    data_test = prep_data.get_test_data()
    
    np.savez(opt.base_path / "tsunami_modeling/data/tsunami_data_normalized.npz", data_train=data_train, data_valid=data_valid, data_test=data_test)
    
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
    # data = np.load(opt.base_path / "tsunami_modeling/data/tsunami_data_normalized.npz", allow_pickle = True)
    # print(data["y"].shape)
    # data_train = data["data_train"].item()
    # data_valid = data["data_valid"].item()
    # data_test = data["data_test"].item()
    log.info("Data loaded.")
    
    data_train = select_space_subset(data_train, opt.num_points_train)
    data_valid = select_space_subset(data_valid, opt.num_points_valid)

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
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=opt.lr_gamma)

    trainer = Trainer(model, optimizer, criterion, \
                      batch_size=opt.batch_size, device=opt.device, lr_scheduler=scheduler, )

    # Train model
    save_path = opt.base_path / opt.model_path 
    log.info("Training.")
    trainer.train(data_train, data_valid, num_epochs=opt.num_epochs, eval_interval=opt.eval_interval, save_path=save_path)
    
    log.info("Training completed.")
    test_error = trainer.test(data_test)
    wandb.log({"test_error": test_error})
    
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