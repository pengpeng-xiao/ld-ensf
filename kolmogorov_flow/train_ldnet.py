import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import wandb
import deepxde as dde

import sys
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, "/work/pengpeng/data-assimilation/")
from src import utils
from src.logger import Logger
from src.normalization import Normalize, Normalize_gaussian
from src.data_preprocess import DataPreprocessor, select_space_subset
from src.fourier_ldnet import ResFourierLDNN
from src.train import Trainer  

def create_training_options():
    parser = argparse.ArgumentParser()
    
    # --------------- path and logging ---------------
    parser.add_argument("--base-path",          type=Path,  default="/work/pengpeng/data-assimilation/kolmogorov_flow")
    parser.add_argument("--data-path",          type=Path,  default="data/data_cplx_Re_500_1500_150x150.npz")
    parser.add_argument("--log-dir",            type=Path,  default="log")
    parser.add_argument("--wandb-entity",       type=str,   default="20307110428",    help="user name of your W&B account")
    parser.add_argument("--wandb-project",      type=str,   default="KF",           help="name of the W&B project")
    parser.add_argument("--name",               type=str,   default="cplx_Re500_1500_150x150_resnet_foureir_10_freeze_gamma_0.5_resd_14", help="name of the run")
    parser.add_argument("--model-path",         type=Path,  default="saved_model/cplx_Re500_1500_150x150_resnet_fourier_10_freeze_gamma_0.5_resd_14")
    
    # --------------- dataset parameter ---------------
    parser.add_argument("--num-points-train",   type=int,   default=5000,   help="number of spatial points in training set")
    parser.add_argument("--num-points-valid",   type=int,   default=5000,   help="number of spatial points in validation set")
    parser.add_argument("--prop-train",         type=float, default=0.6)
    parser.add_argument("--prop-valid",         type=float, default=0.2)  
    parser.add_argument("--prop-test",          type=float, default=0.2) 
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
    parser.add_argument("--const-u",            action="store_true",  default=False)
    
    # --------------- training parameter ---------------
    parser.add_argument("--device",             type=str,   default="cuda:0")
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--batch-size",         type=int,   default=6)
    parser.add_argument("--learning-rate",      type=float, default=1e-3)
    parser.add_argument("--num-epochs",         type=int,   default=2000)
    parser.add_argument("--eval-interval",      type=int,   default=20)
    parser.add_argument("--lr-gamma",           type=float, default=0.5,    help="learning rate decay factor")

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

    # dataset["u"] = np.tile(dataset["u"][:, None, None], (1, dataset["y"].shape[1], 1))
    dataset["u"] = dataset["u"][:, None]
    
    
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
    
    data_train_y  = data_train["y"]
    mean = np.mean(data_train_y, axis=tuple(range(data_train_y.ndim-1)))
    std = np.std(data_train_y, axis=tuple(range(data_train_y.ndim-1)))
    np.savez(opt.base_path / opt.model_path / "mean_std.npz", mean=mean, std=std)
    normalize_y = Normalize_gaussian(mean, std)
    data_train["y"] = normalize_y.normalize_forw(data_train_y)
    data_valid["y"] = normalize_y.normalize_forw(data_valid["y"])
    data_test["y"] = normalize_y.normalize_forw(data_test["y"])
    
    return data_train, data_valid, data_test

def main(opt):
    log = Logger(log_dir=opt.base_path / opt.model_path / opt.log_dir)
    log.info("=======================================================")
    log.info("           Kolmogorov Flow Data Assimilation           ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")
    
    wandb.init(entity=opt.wandb_entity, project=opt.wandb_project, name=opt.name, 
               dir=str(opt.base_path / opt.model_path / opt.log_dir), config=vars(opt), save_code=True)
    
    utils.set_seed(opt.seed) 

    data_train, data_valid, data_test = load_data(opt)
    log.info("Data loaded.")
    
    data_train = select_space_subset(data_train, opt.num_points_train)
    data_valid = select_space_subset(data_valid, opt.num_points_valid)

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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=opt.lr_gamma)

    trainer = Trainer(model, optimizer, criterion, \
                      batch_size=opt.batch_size, device=opt.device, lr_scheduler=scheduler, )
    
    try: # Train model
        trainer.train(data_train, data_valid, num_epochs=opt.num_epochs, 
                    eval_interval=opt.eval_interval, save_path=opt.base_path / opt.model_path)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA out of memory. Trying to clear cache.")
            torch.cuda.empty_cache()
        else:
            raise e
    
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