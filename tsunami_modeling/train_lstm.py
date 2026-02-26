import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb 

import shutil
from pathlib import Path

from src.encoder import *
# set seed
torch.manual_seed(0)

def main(device, epochs, data_path, hidden_size, num_layers, dropout, save_path, epsilon):
    data = torch.load(data_path, map_location="cpu")
    input = data["data_train"]["observation"].reshape(120, 51, -1).to(device)
    # print("data['data_train']['u'] shape", data["data_train"]["u"].shape)
    u = data["data_train"]["u"].to(device)
    label = data["data_train"]["latent_states"].to(device)
    label = torch.concatenate((label, u), dim = 2).to(device)
    label = label.to(device) * epsilon #.reshape(latents.shape[0] * latents.shape[1], -1)
    
    # repeat the first time slice several times
    # slice_input = input[:, 0, :].unsqueeze(1).repeat((1, 4, 1))
    # slice_label = label[:, 0, :].unsqueeze(1).repeat((1, 4, 1))
    # input = torch.cat((slice_input, input), dim = 1)
    # label = torch.cat((slice_label, label), dim = 1)

    input_val = data["data_valid"]["observation"].reshape(40, 51, -1).to(device)
    label_val = data["data_valid"]["latent_states"]
    label_val = label_val.to(device)
    u = data["data_valid"]["u"].to(device)

    label_val = torch.concatenate((label_val, u), dim = 2).to(device)
    label_val = label_val.to(device) * epsilon #.reshape(latents.shape[0] * latents.shape[1], -1)
    
    # slice_input = input_val[:, 0, :].unsqueeze(1).repeat((1, 4, 1))
    # slice_label = label_val[:, 0, :].unsqueeze(1).repeat((1, 4, 1))
    # input_val = torch.cat((slice_input, input_val), dim = 1)
    # label_val = torch.cat((slice_label, label_val), dim = 1)
    
    model = TimeSeriesLSTM(input.shape[-1], hidden_size, label.shape[-1], num_layers=num_layers, dropout=dropout,) 
    # model = TimeSeriesRNN(input.shape[-1], hidden_size, label.shape[-1], num_layers=num_layers,) 
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    T_max = 5000
    # scheduler = CosineAnnealingLR(optimizer, T_max, eta_min=0.001) #
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    model.to(device)

    # scaler = torch.cat((torch.tensor([10,5,3]), torch.ones(48)))[None, :, None].to(device)
    # training loop
    for epoch in range(epochs):
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")

            # validation
            with torch.no_grad():
                model.eval()
                output_val = model(input_val)
                loss_val = criterion(output_val, label_val)
                rmse = torch.sqrt(torch.mean((output_val/epsilon - label_val/epsilon) ** 2)) / torch.sqrt(torch.mean((label_val/epsilon) ** 2))
                rmse_s = torch.sqrt(torch.mean((output_val[:,:,:10]/epsilon - label_val[:,:,:10]/epsilon) ** 2)) / torch.sqrt(torch.mean((label_val[:,:,:10]/epsilon) ** 2))
                rmse_u = torch.sqrt(torch.mean((output_val[:,:,10:]/epsilon - label_val[:,:,10:]/epsilon) ** 2)) / torch.sqrt(torch.mean((label_val[:,:,10:]/epsilon) ** 2))
                wandb.log({"loss": loss.item(), "val_loss": loss_val.item(), "error total": rmse, 
                          "error s": rmse_s, "error u": rmse_u}, step=epoch)
                model.train()
    # print(output_val)
                
                
    torch.save(model, Path(save_path) / "lstm.ckpt")
    wandb.save(str(Path(save_path) / "lstm.ckpt"))
    
    # current_file = __file__
    # destination_file = Path(save_path) / "train_lstm.py"
    # shutil.copyfile(current_file, destination_file)
    
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    device = torch.device("cuda:3")
    epochs = 20000
    hidden_size = 256
    num_layers = 1
    dropout = 0.1
    epsilon = 1.0
    data_path = "./data/data_2000_150x150_coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u.pth"
    save_path = "./saved_model/coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u"
    config={
        "device": str(device),
        "epochs": epochs,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": 1e-4,
    }
    wandb.init(entity="20307110428", project="SW_KF_lstm", name="coriolis_recd_10_w_300_lr_1e-3_bs_2_ic_nor_u", 
            dir=save_path, config=config, save_code=True)
    main(device, epochs, data_path, hidden_size, num_layers, dropout, save_path, epsilon)
