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
    # print(data["data_train"]["observation"].shape)
    # print(data["data_train"]["latent_states"].shape)
    # print(data["data_train"]["u"].shape)
    input = data["data_train"]["observation"].reshape(120, 63, -1).to(device)
    u = data["data_train"]["u"].unsqueeze(1).repeat((1, 63, 1)).to(device)
    label = data["data_train"]["latent_states"].to(device)
    label = torch.concatenate((label, u), dim = 2).to(device)
    label = label.to(device) #* epsilon #.reshape(latents.shape[0] * latents.shape[1], -1)

    input_val = data["data_valid"]["observation"].reshape(40, 63, -1).to(device)
    label_val = data["data_valid"]["latent_states"].to(device)
    u = data["data_valid"]["u"].unsqueeze(1).repeat((1, 63, 1)).to(device)

    label_val = torch.concatenate((label_val, u), dim = 2).to(device)
    label_val = label_val.to(device) #* epsilon #.reshape(latents.shape[0] * latents.shape[1], -1)
    model = TimeSeriesLSTM(input.shape[-1], hidden_size, label.shape[-1], num_layers=num_layers, dropout=dropout,) 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    T_max = 5000
    scheduler = CosineAnnealingLR(optimizer, T_max, eta_min=0.002) 
    model.to(device)

    # training loop
    for epoch in range(epochs):
        output = model(input)
        # loss_s = criterion(output, label)
        loss_s = criterion(output[:,:,:50], label[:,:,:50])
        loss_u = criterion(output[:,:,50:], label[:,:,50:])
        loss = loss_s + loss_u
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
                # loss_val = criterion(output_val, label_val)
                loss_val_s = criterion(output_val[:,:,:50], label_val[:,:,:50])
                loss_val_u = criterion(output_val[:,:,50:], label_val[:,:,50:])
                loss_val = loss_val_s + loss_val_u
                rmse = torch.sqrt(torch.mean((output_val/epsilon - label_val/epsilon) ** 2)) / torch.sqrt(torch.mean((label_val/epsilon) ** 2))
                rmse_s = torch.sqrt(torch.mean((output_val[:,:,:50]/epsilon - label_val[:,:,:50]/epsilon) ** 2)) / torch.sqrt(torch.mean((label_val[:,:,:50]/epsilon) ** 2))
                rmse_u = torch.sqrt(torch.mean((output_val[:,:,50:]/epsilon - label_val[:,:,50:]/epsilon) ** 2)) / torch.sqrt(torch.mean((label_val[:,:,50:]/epsilon) ** 2))
                wandb.log({"loss": loss.item(), "val_loss": loss_val.item(), "error total": rmse, 
                          "error s": rmse_s, "error u": rmse_u}, step=epoch)
                model.train()
    # print(output_val)
                
    torch.save(model, Path(save_path) / "lstm_random_32.ckpt")
    wandb.save(str(Path(save_path) / "lstm_random_32.ckpt"))
    
    # current_file = __file__
    # destination_file = Path(save_path) / "train_lstm_1.py"
    # shutil.copyfile(current_file, destination_file)
    
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    device = torch.device("cuda:0")
    epochs = 80000
    hidden_size = 512
    num_layers = 1
    dropout = 0
    epsilon = 1
    # TODO: Update these paths to match your environment
    data_path = "saved_model/parameter_shuf_t21d/observation_random_32.pth"
    save_path = "saved_model/parameter_shuf_t21d"
    config={
        "device": str(device),
        "epochs": epochs,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": 1e-4,
    }
    wandb.init(entity="20307110428", project="SW_KF_lstm", name="parameter_shuf_t21d",
            dir=save_path, config=config, save_code=True)
    main(device, epochs, data_path, hidden_size, num_layers, dropout, save_path, epsilon)