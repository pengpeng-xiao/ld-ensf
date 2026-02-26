import torch
import torch.nn as nn
import wandb 

import shutil
from pathlib import Path

from src.encoder import *
# set seed
torch.manual_seed(0)

def main(device, epochs, data_path, hidden_size, num_layers, dropout, save_path, epsilon):
    data = torch.load(data_path, map_location="cpu")
    input = data["data_train"]["observation"].reshape(120, 51, -1).to(device)
    u = data["data_train"]["u"].to(device)
    label = data["data_train"]["latent_states"].to(device)
    label = torch.concatenate((label, u), dim = 2).to(device)
    label = label.to(device) * epsilon #.reshape(latents.shape[0] * latents.shape[1], -1)

    input_val = data["data_valid"]["observation"].reshape(40, 51, -1).to(device)
    label_val = data["data_valid"]["latent_states"]
    label_val = label_val.to(device)
    u = data["data_valid"]["u"].to(device)

    label_val = torch.concatenate((label_val, u), dim = 2).to(device)
    label_val = label_val.to(device) * epsilon #.reshape(latents.shape[0] * latents.shape[1], -1)
    
    model = TimeSeriesLSTM(input.shape[-1], hidden_size, label.shape[-1], num_layers=num_layers, dropout=dropout,) 
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    model.to(device)

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
       
    torch.save(model, Path(save_path) / "lstm.ckpt")
    wandb.save(str(Path(save_path) / "lstm.ckpt"))
    
    print(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    device = torch.device("cuda:1")
    epochs = 20000
    hidden_size = 256
    num_layers = 1
    dropout = 0.1
    epsilon = 1.0
    data_path = "tsunami_modeling/data/observation_data.pth"
    save_path = "tsunami_modeling/saved_model/ldnet"
    config={
        "device": str(device),
        "epochs": epochs,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": 1e-4,
    }
    wandb.init(entity=None, project=None, name=None, 
            dir=save_path, config=config, save_code=True)
    main(device, epochs, data_path, hidden_size, num_layers, dropout, save_path, epsilon)
