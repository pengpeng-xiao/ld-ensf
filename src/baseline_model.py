import torch
import torch.linalg as linalg
import deepxde as dde
import numpy as np
import numpy.linalg as nla
import numpy.random as rnd
from .encoder import TimeSeriesLSTM
from .model import LDNN, ResNN, FourierEmbedding, FourierRec

parallel = 40


class Rec_Only(dde.nn.NN):
    """Reconstruction-only model without latent dynamics.

    Data should be a dictionary with keys: y, u, x, dt.
    x must have shape (Ni, Nt, Nx, dimx).
    """
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
        self.rec = ResNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
    
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        Ni = x.shape[0]
        Nt = x.shape[1]
        Nx = x.shape[2]
        dim_x = x.shape[-1]
        dim_u = u.shape[-1]
        time = torch.arange(Nt, device=device) * dt
        time = time[None,:,None,None].expand(Ni,Nt,Nx,1)
        if len(u.shape) == 2:
            u = u[:,None,None,:].expand(Ni,Nt,Nx,dim_u)
        elif len(u.shape) == 3:
            u = u[:,:,None,:].expand(Ni,Nt,Nx,dim_u)

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

        return x_rec

class Rec_Only_fourier(dde.nn.NN):
    """Fourier-based reconstruction-only model without latent dynamics.

    Data should be a dictionary with keys: y, u, x, dt.
    x must have shape (Ni, Nt, Nx, dimx).
    """

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
        self.rec = ResNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
    
    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # here we suppose x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        Ni = x.shape[0]
        Nt = x.shape[1]
        Nx = x.shape[2]
        dim_x = x.shape[-1]
        dim_u = u.shape[-1]
        time = torch.arange(Nt, device=device) * dt
        time = time[None,:,None,None].expand(Ni,Nt,Nx,1)
        if len(u.shape) == 2:
            u = u[:,None,None,:].expand(Ni,Nt,Nx,dim_u)
        else:
            u = u[:,:,None,:].expand(Ni,Nt,Nx,dim_u)

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

        return x_rec