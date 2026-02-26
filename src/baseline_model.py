import torch
import torch.linalg as linalg
import deepxde as dde
import numpy as np
import numpy.linalg as nla
import numpy.random as rnd
from .encoder import TimeSeriesLSTM
from .resnet import ResNN
from .fourier_ldnet import FourierEmbedding, FourierRec
from .model import LDNN

parallel = 40

# scaling = 500
# obs_sigma =  0.1
# eps_alpha = 0.05
# euler_steps = 100

# enkf_scaling = 1
# enkf_obs_sigma = 0
        
'''
The data should be a dictionary with the following keys:
y, u, x, dt, (y_x, y_u)
x must has shape (Ni, Nt, Nx, dimx). 
'''
class Rec_Only(dde.nn.NN):
    # Suppose x must has shape (Ni, Nt, Nx, dimx). Output the latent state history.
    def __init__(
        self,
        layer_sizes_rec,
        activation,
        kernel_initializer,
        dropout=0,
    ):
        super().__init__()
        self.activation_rec = activation
        self.dropout = dropout
        
        # dyn_input_size = layer_sizes_dyn[0]  # input size for the dynamic network
        # dyn_hidden_size = layer_sizes_dyn[1]  # hidden size for the dynamic network
        # dyn_output_size = layer_sizes_dyn[-1]  # output size for the dynamic network
        # dyn_num_layers = len(layer_sizes_dyn) - 2  # number of layers for the dynamic network
        # # Fully connected network
        # # self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
        # self.dyn = TimeSeriesLSTM(dyn_input_size, dyn_hidden_size, dyn_output_size, dyn_num_layers, dropout=dropout)
        self.rec = ResNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        # self.num_latent_states = layer_sizes_dyn[-1]
        # self.state = None
        # self.state_history = []
    
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        # if latent_init is False:
        #     self.state = torch.zeros(u.shape[0],self.num_latent_states).to(device)
        # else:
        #     self.state = data["latent"]
                       
        # self.state_history = []
        # for i in range(x.shape[1]):
        # dyn net to encode the input function
        # if len(u.shape) == 2:
        #     u_ti = u
        # else:
        #     u_ti = u[:,i,:] # if u has shape (Ni, Nt, dimu)
        #     input = torch.cat((u_ti,self.state),dim=1)
        #     if equilibrium:
        #         self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
        #     else:
        #         self.state = self.state + dt * self.dyn(input)
        #     self.state_history.append(self.state)

        # self.state_history = torch.stack(self.state_history).transpose(0, 1)
        
        # if latent_state:
        #     return self.state_history
        
        Ni = x.shape[0]
        Nt = x.shape[1]
        Nx = x.shape[2]
        dim_x = x.shape[-1]
        dim_u = u.shape[-1]
        # dims = self.state_history.shape[2]
        # self.state_history_ = self.state_history.unsqueeze(2).expand(Ni,Nt,Nx,dims)
        time = torch.arange(Nt, device=device) * dt
        time = time[None,:,None,None].expand(Ni,Nt,Nx,1)
        if len(u.shape) == 2:
            u = u[:,None,None,:].expand(Ni,Nt,Nx,dim_u)
        elif len(u.shape) == 3:
            u = u[:,:,None,:].expand(Ni,Nt,Nx,dim_u)
        # concatenate latent state and space
        if self.training:
            x_rec = torch.cat([x, time, u],dim=-1)
            x_rec = self.rec(x_rec)
        else:
            x_rec = []
            with torch.no_grad():
                x_rec_input = torch.cat([x, time, u],dim=-1).detach()
                for i in np.arange(0,self.ensemble_size,parallel):
                    x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=0)
                # print(x_rec.device)
        
        # if latent_state:
        #     return self.state_history
        # else:
        return x_rec

class Rec_Only_fourier(dde.nn.NN):
    # Suppose x must has shape (Ni, Nt, Nx, dimx). Output the latent state history.
    def __init__(
        self,
        layer_sizes_rec,
        activation,
        kernel_initializer,
        dropout=0,
    ):
        super().__init__()
        self.activation_rec = activation
        self.dropout = dropout
        
        # dyn_input_size = layer_sizes_dyn[0]  # input size for the dynamic network
        # dyn_hidden_size = layer_sizes_dyn[1]  # hidden size for the dynamic network
        # dyn_output_size = layer_sizes_dyn[-1]  # output size for the dynamic network
        # dyn_num_layers = len(layer_sizes_dyn) - 2  # number of layers for the dynamic network
        # # Fully connected network
        # # self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
        # self.dyn = TimeSeriesLSTM(dyn_input_size, dyn_hidden_size, dyn_output_size, dyn_num_layers, dropout=dropout)
        self.rec = ResNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        # self.num_latent_states = layer_sizes_dyn[-1]
        # self.state = None
        # self.state_history = []
    
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        # if latent_init is False:
        #     self.state = torch.zeros(u.shape[0],self.num_latent_states).to(device)
        # else:
        #     self.state = data["latent"]
                       
        # self.state_history = []
        # for i in range(x.shape[1]):
        # dyn net to encode the input function
        # if len(u.shape) == 2:
        #     u_ti = u
        # else:
        #     u_ti = u[:,i,:] # if u has shape (Ni, Nt, dimu)
        #     input = torch.cat((u_ti,self.state),dim=1)
        #     if equilibrium:
        #         self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
        #     else:
        #         self.state = self.state + dt * self.dyn(input)
        #     self.state_history.append(self.state)

        # self.state_history = torch.stack(self.state_history).transpose(0, 1)
        
        # if latent_state:
        #     return self.state_history
        
        Ni = x.shape[0]
        Nt = x.shape[1]
        Nx = x.shape[2]
        dim_x = x.shape[-1]
        dim_u = u.shape[-1]
        # dims = self.state_history.shape[2]
        # self.state_history_ = self.state_history.unsqueeze(2).expand(Ni,Nt,Nx,dims)
        time = torch.arange(Nt, device=device) * dt
        time = time[None,:,None,None].expand(Ni,Nt,Nx,1)
        if len(u.shape) == 2:
            u = u[:,None,None,:].expand(Ni,Nt,Nx,dim_u)
        else:
            u = u[:,:,None,:].expand(Ni,Nt,Nx,dim_u)
        # concatenate latent state and space
        if self.training:
            x_rec = torch.cat([x, time, u],dim=-1)
            x_rec = self.rec(x_rec)
        else:
            x_rec = []
            with torch.no_grad():
                x_rec_input = torch.cat([x, time, u],dim=-1).detach()
                for i in np.arange(0,self.ensemble_size,parallel):
                    x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=0)
                # print(x_rec.device)
        
        # if latent_state:
        #     return self.state_history
        # else:
        return x_rec