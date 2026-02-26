import time

import torch
import torch.linalg as linalg
import deepxde as dde
import numpy as np
import numpy.linalg as nla
import numpy.random as rnd
from .encoder import *

parallel = 10


# ---------------------------------------------------------------------------
# Helper / building-block modules
# ---------------------------------------------------------------------------

class ActivationModule(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)


class ResidualBlock1d(nn.Module):
    """
    input_shape = (batch_size, input_dim)
    output_shape = (batch_size, output_dim)
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str,
                 kernel_initializer: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation1 = activations.get(activation)
        self.activation2 = activations.get(activation)
        self.init_func = initializers.get(kernel_initializer)

        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        self.linear2 = nn.Linear(self.output_dim, self.output_dim)
        self.shortcut = nn.Sequential()
        if self.input_dim != self.output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim),
            )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_func(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.activation1(self.linear1(x))
        y = self.linear2(y)
        y = self.activation2(y + self.shortcut(x))
        return y


class ResNN(nn.Module):
    """
    input_shape = (batch_size, input_dim)
    output_shape = (batch_size, output_dim)
    """

    def __init__(self,
                 layer_sizes: list,
                 activation: str,
                 kernel_initialier: str):
        super().__init__()
        self.hidden_depth = len(layer_sizes) - 2
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]
        self.hidden_dim = layer_sizes[1]  # suppose the hidden layers have the same dimension

        self.activation = activations.get(activation)
        self.init_func = initializers.get(kernel_initialier)

        self.num_residual_blocks = (self.hidden_depth - 1) // 2
        self.num_remaining_hidden_layers = (self.hidden_depth - 1) % 2

        self.residual_blocks = nn.ModuleList()
        for _ in range(self.num_residual_blocks):
            self.residual_blocks.append(ResidualBlock1d(self.hidden_dim,
                                                        self.hidden_dim,
                                                        activation,
                                                        kernel_initialier))

        if self.num_remaining_hidden_layers == 1:
            layers = [nn.Linear(self.hidden_dim, self.hidden_dim),
                      ActivationModule(activations.get(activation))]
            self.remaining_hidden_layers = nn.ModuleList(layers)
        else:
            self.remaining_hidden_layers = nn.ModuleList()

        self.input_hidden_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden_output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.init_params()

    def init_params(self):
        for block in self.residual_blocks:
            block.init_params()
        for layer in self.remaining_hidden_layers:
            if isinstance(layer, nn.Linear):
                self.init_func(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        self.init_func(self.input_hidden_linear.weight)
        if self.input_hidden_linear.bias is not None:
            nn.init.zeros_(self.input_hidden_linear.bias)
        self.init_func(self.hidden_output_linear.weight)
        if self.hidden_output_linear.bias is not None:
            nn.init.zeros_(self.hidden_output_linear.bias)

    def forward(self, x):
        x = self.activation(self.input_hidden_linear(x))
        for block in self.residual_blocks:
            x = block(x)
        for layer in self.remaining_hidden_layers:
            x = layer(x)
        x = self.hidden_output_linear(x)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Cell representing a generic Fourier embedding (i.e. a 2D matrix representing an encoding).
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.encoding = torch.nn.Linear(in_feats, out_feats, bias=False)
        self.encoding.weight.requires_grad = False

    def forward(self, inp):
        return self.encoding(inp)


class FourierRec(dde.nn.NN):
    def __init__(self, fourier_embedding, rec):
        super().__init__()
        self.fourier_embedding = fourier_embedding
        self.rec = rec

    def forward(self, x, state_history):
        points_projected = self.fourier_embedding(2 * torch.pi * x)
        x_rec_input = torch.cat([torch.sin(points_projected), torch.cos(points_projected), state_history], dim=-1)
        return self.rec(x_rec_input)


# ---------------------------------------------------------------------------
# Latent-dynamics neural network models
# Data dict keys: y, u, x, dt, (y_x, y_u)
# x must have shape (Ni, Nt, Nx, dimx).
# ---------------------------------------------------------------------------

class LDNN(dde.nn.NN):
    def __init__(
        self,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_dyn = dde.nn.activations.get(activation["dyn"])
            self.activation_rec = dde.nn.activations.get(activation["rec"])
        else:
            activation_dyn = self.activation_rec = dde.nn.activations.get(activation)
        if callable(layer_sizes_dyn[1]):
            self.dyn = layer_sizes_dyn[1]
        else:
            self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
        self.rec = dde.nn.FNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        self.num_latent_states = layer_sizes_dyn[-1]
        self.state = None
        self.state_history = []

    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        if latent_init is False:
            self.state = torch.zeros(u.shape[0], self.num_latent_states).to(device)
        else:
            self.state = data["latent"]

        self.state_history = []
        for i in range(x.shape[1]):
            if len(u.shape) == 2:
                u_ti = u
            else:
                u_ti = u[:, i, :]  # u has shape (Ni, Nt, dimu)
            input = torch.cat((u_ti, self.state), dim=1)
            if equilibrium:
                self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
            else:
                self.state = self.state + dt * self.dyn(input)
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)

        if latent_state:
            return self.state_history

        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni, Nt, Nx, dims)
        if self.training:
            x_rec = torch.cat([x, self.state_history_], dim=-1)
            x_rec = self.rec(x_rec)
        else:
            x_rec = []
            with torch.no_grad():
                x_rec_input = torch.cat([x.cpu(), self.state_history.cpu()], dim=-1).detach()
                for i in np.arange(0, Nx, parallel):
                    x_rec_i = self.rec(x_rec_input[:, :, i:i+parallel].to(device)).detach().cpu()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=2)

        return x_rec

    def reverse_SDE(self, x0, score_likelihood=None, time_steps=100, save_path=False, device="cpu", eps_alpha=0.05, scaling=500):
        ensemble_size = x0.shape[0]

        def cond_alpha(t):
            return 1 - (1 - eps_alpha) * t

        def cond_sigma_sq(t):
            eps_beta = 0
            return eps_beta + t * (1 - eps_beta)

        def f_func(t):
            # f = d(log_alpha)/dt
            alpha_t = cond_alpha(t)
            return -(1 - eps_alpha) / alpha_t

        def g_sq(t):
            # g = d(sigma_t^2)/dt - 2f * sigma_t^2
            return 1 - 2 * f_func(t) * cond_sigma_sq(t)

        def g_func(t):
            return np.sqrt(g_sq(t))

        dt = 1.0 / time_steps
        xt = torch.randn(ensemble_size, x0.shape[1], device=device)
        t = 1.0

        if save_path:
            path_all = [xt]
            t_vec = [t]

        for i in range(time_steps):
            alpha_t = cond_alpha(t)
            sigma2_t = cond_sigma_sq(t)
            diffuse = g_func(t)

            if score_likelihood is not None:
                xt += -dt * (f_func(t) * xt + diffuse**2 * ((xt - alpha_t * x0) / sigma2_t) - diffuse**2 * score_likelihood(xt, t)) \
                    + np.sqrt(dt) * diffuse * torch.randn_like(xt)
            else:
                xt += -dt * (f_func(t) * xt + diffuse**2 * ((xt - alpha_t * x0) / sigma2_t)) + np.sqrt(dt) * diffuse * torch.randn_like(xt)

            if save_path:
                path_all.append(xt)
                t_vec.append(t)

            t = t - dt

        if save_path:
            return path_all, t_vec
        else:
            return xt

    def forward_data_assimilation_ensf(self, data, observations, encoder, noise_level, device, equilibrium=False, latent_state=False, optimal_latent=None,
                                       scaling=500, obs_sigma=0.1, eps_alpha=0.05, euler_steps=100):
        # x has shape (Ni, Nt, Nx, dimx), observations have shape (Nt, Nx, dimx), u has shape (2)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        assert len(u.shape) == 2

        observation_vector = observations.reshape([observations.shape[0], -1])
        observation_vector = observation_vector + torch.randn_like(observation_vector) * noise_level * observation_vector
        encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        self.state = torch.zeros(u.shape[0], self.num_latent_states).to(device)
        self.state_history, u_history = [], []

        for i in range(x.shape[1]):
            if i == 0 and optimal_latent is not None:
                obs_ti = torch.cat((u, optimal_latent), dim=1)
            else:
                obs_ti = encoded_obs[i, :]
            scaled_obs_ti = obs_ti * scaling
            input = torch.cat((u, self.state), dim=1)

            with torch.no_grad():
                if equilibrium:
                    self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
                else:
                    self.state = self.state + dt * self.dyn(input)

            def g_tau(t):
                return 1 - t

            def score_likelihood(xt, t):
                # analytical ∇z log P(Yt+1|z)
                score_x = -(xt - scaled_obs_ti) / obs_sigma**2
                return g_tau(t) * score_x

            latent = torch.concatenate((self.state.view(self.state.shape[0], -1), u), dim=1)
            post_state = self.reverse_SDE(x0=latent * scaling, score_likelihood=score_likelihood,
                                          time_steps=euler_steps, device=device, eps_alpha=eps_alpha,
                                          scaling=scaling) / scaling
            self.state = post_state[:, :self.state.shape[1]]
            u = post_state[:, self.state.shape[1]:]
            self.state_history.append(self.state)
            u_history.append(u)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        u_history = torch.stack(u_history).transpose(0, 1)

        if latent_state:
            return self.state_history, u_history, encoded_obs

        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history = self.state_history.unsqueeze(2).expand(Ni, Nt, Nx, dims)

        x_rec = []
        with torch.no_grad():
            x_rec_input = torch.cat([x.cpu(), self.state_history.cpu()], dim=-1).detach()
            for i in np.arange(0, Nx, parallel):
                x_rec_i = self.rec(x_rec_input[:, :, i:i+parallel].to(device)).detach().cpu()
                x_rec.append(x_rec_i)
            x_rec_tensor = torch.cat(x_rec, dim=2)
        return x_rec_tensor

    def forward_data_assimilation_ensf_sparse_time(self, data, observations, encoder, noise_level, device, equilibrium=False, latent_state=False, optimal_latent=None,
                                                    scaling=500, obs_sigma=0.1, eps_alpha=0.05, euler_steps=100, interval=5):
        # x has shape (Ni, Nt, Nx, dimx), observations have shape (Nt, Nx, dimx), u has shape (2)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        assert len(u.shape) == 2

        observation_vector = observations.reshape([observations.shape[0], -1])
        observation_vector = observation_vector + torch.randn_like(observation_vector) * noise_level
        encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        self.state = torch.zeros(x.shape[0], self.num_latent_states).to(device)
        self.state_history, u_history = [], []

        for i in range(x.shape[1]):
            if i == 0 and optimal_latent is not None:
                obs_ti = torch.cat((u, optimal_latent), dim=1)
            else:
                obs_ti = encoded_obs[i // interval, :]
            scaled_obs_ti = obs_ti * scaling
            input = torch.cat((u, self.state), dim=1)

            with torch.no_grad():
                if equilibrium:
                    self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
                else:
                    self.state = self.state + dt * self.dyn(input)

            def g_tau(t):
                return 1 - t

            def score_likelihood(xt, t):
                # analytical ∇z log P(Yt+1|z)
                score_x = -(xt - scaled_obs_ti) / obs_sigma**2
                return g_tau(t) * score_x

            if i == 0:
                latent = torch.concatenate((self.state.view(self.state.shape[0], -1), u), dim=1)
                post_state = self.reverse_SDE(x0=latent * scaling, score_likelihood=score_likelihood,
                                              time_steps=euler_steps, device=device, eps_alpha=eps_alpha) / scaling
                self.state = post_state[:, :self.state.shape[1]]
                u = post_state[:, self.state.shape[1]:]
                u_history.append(u)
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        u_history = torch.stack(u_history).transpose(0, 1)

        if latent_state:
            return self.state_history, u_history, encoded_obs

        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history = self.state_history.unsqueeze(2).expand(Ni, Nt, Nx, dims)

        x_rec = []
        with torch.no_grad():
            x_rec_input = torch.cat([x, self.state_history], dim=-1).detach()
            for i in np.arange(0, self.ensemble_size, parallel):
                x_rec_i = self.rec(x_rec_input[i:i+parallel]).detach().cpu()
                x_rec.append(x_rec_i)
            x_rec_tensor = torch.cat(x_rec, dim=0)
        return x_rec_tensor


class ResLDNN(LDNN):
    def __init__(
        self,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
    ):
        super().__init__(layer_sizes_dyn, layer_sizes_rec, activation, kernel_initializer)
        if isinstance(activation, dict):
            activation_dyn = dde.nn.activations.get(activation["dyn"])
            self.activation_rec = dde.nn.activations.get(activation["rec"])
        else:
            activation_dyn = self.activation_rec = dde.nn.activations.get(activation)
        if callable(layer_sizes_dyn[1]):
            self.dyn = layer_sizes_dyn[1]
        else:
            self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)
        self.rec = ResNN(layer_sizes_rec, self.activation_rec, kernel_initializer)
        self.num_latent_states = layer_sizes_dyn[-1]
        self.state = None
        self.state_history = []

    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        if latent_init is False:
            self.state = torch.zeros(u.shape[0], self.num_latent_states).to(device)
        else:
            self.state = data["latent"]

        self.state_history = []
        for i in range(x.shape[1]):
            if len(u.shape) == 2:
                u_ti = u
            else:
                u_ti = u[:, i, :]  # u has shape (Ni, Nt, dimu)
            input = torch.cat((u_ti, self.state), dim=1)
            dyn_output = self.dyn(input)
            self.state = self.state + dt * dyn_output
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)

        if latent_state:
            return self.state_history

        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni, Nt, Nx, dims)
        if self.training:
            x_rec = torch.cat([x, self.state_history_], dim=-1)
            x_rec = self.rec(x_rec)
        else:
            x_rec = []
            with torch.no_grad():
                x_rec_input = torch.cat([x.cpu(), self.state_history_.cpu()], dim=-1).detach()
            for i in np.arange(0, Nx, parallel):
                x_rec_i = self.rec(x_rec_input[:, :, i:i+parallel].to(device)).detach().cpu()
                x_rec.append(x_rec_i)
            x_rec = torch.cat(x_rec, dim=2)

        return x_rec


class FourierLDNN(dde.nn.NN):
    def __init__(
        self,
        fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
        dropout=0
    ):
        super().__init__()
        self.fourier_mapping_size = fourier_mapping_size
        self.n_coords = layer_sizes_rec[0] - layer_sizes_dyn[-1]

        self.B = FourierEmbedding(self.n_coords, fourier_mapping_size)
        layer_sizes_rec_f = [layer_sizes_dyn[-1] + 2 * self.fourier_mapping_size] + layer_sizes_rec[1:]

        if isinstance(dropout, list):
            self.dyn = FNN(layer_sizes_dyn, activation, kernel_initializer, dropout=dropout[0])
            self.rec = FourierRec(self.B, FNN(layer_sizes_rec_f, activation, kernel_initializer, dropout=dropout[1]))
        else:
            self.dyn = FNN(layer_sizes_dyn, activation, kernel_initializer, dropout=dropout)
            self.rec = FourierRec(self.B, FNN(layer_sizes_rec_f, activation, kernel_initializer, dropout=dropout))

        self.num_latent_states = layer_sizes_dyn[-1]
        self.state = None
        self.state_history = []

    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        if latent_init is False:
            self.state = torch.zeros(u.shape[0], self.num_latent_states).to(device)
        else:
            self.state = data["latent"]

        self.state_history = []
        for i in range(x.shape[1]):
            if len(u.shape) == 2:
                u_ti = u
            else:
                u_ti = u[:, i, :]  # u has shape (Ni, Nt, dimu)
            input = torch.cat((u_ti, self.state), dim=1)
            if equilibrium:
                self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
            else:
                self.state = self.state + dt * self.dyn(input)
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)

        if latent_state:
            return self.state_history

        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni, Nt, Nx, dims)
        if self.training:
            x_rec = self.rec(x, self.state_history_)
        else:
            x_rec = []
            with torch.no_grad():
                x_rec_input = self.state_history_.detach()
                for i in np.arange(0, self.ensemble_size, parallel):
                    x_rec_i = self.rec(x[i:i+parallel], x_rec_input[i:i+parallel]).detach()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=0)

        return x_rec


class ResFourierLDNN(LDNN):
    def __init__(
        self,
        fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation,
        kernel_initializer,
    ):
        super().__init__(layer_sizes_dyn, layer_sizes_rec, activation, kernel_initializer)
        if isinstance(activation, dict):
            activation_dyn = dde.nn.activations.get(activation["dyn"])
            self.activation_rec = dde.nn.activations.get(activation["rec"])
        else:
            activation_dyn = self.activation_rec = dde.nn.activations.get(activation)
        if callable(layer_sizes_dyn[1]):
            self.dyn = layer_sizes_dyn[1]
        else:
            self.dyn = dde.nn.FNN(layer_sizes_dyn, activation_dyn, kernel_initializer)

        self.fourier_mapping_size = fourier_mapping_size
        self.n_coords = layer_sizes_rec[0] - layer_sizes_dyn[-1]

        self.B = FourierEmbedding(self.n_coords, fourier_mapping_size)
        layer_sizes_rec_f = [layer_sizes_dyn[-1] + 2 * self.fourier_mapping_size] + layer_sizes_rec[1:]
        self.rec = FourierRec(self.B, ResNN(layer_sizes_rec_f, self.activation_rec, kernel_initializer))

        self.num_latent_states = layer_sizes_dyn[-1]
        self.state = None
        self.state_history = []

    def forward(self, data, device, equilibrium=False, latent_state=False, latent_init=False):
        # x has shape (Ni, Nt, Nx, dimx)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        if latent_init is False:
            self.state = torch.zeros(u.shape[0], self.num_latent_states).to(device)
        else:
            self.state = data["latent"]

        self.state_history = []
        for i in range(x.shape[1]):
            if len(u.shape) == 2:
                u_ti = u
            else:
                u_ti = u[:, i, :]  # u has shape (Ni, Nt, dimu)
            input = torch.cat((u_ti, self.state), dim=1)
            if equilibrium:
                self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
            else:
                self.state = self.state + dt * self.dyn(input)
            self.state_history.append(self.state)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)

        if latent_state:
            return self.state_history

        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni, Nt, Nx, dims)
        if self.training:
            x_rec = self.rec(x, self.state_history_)
        else:
            x_rec = []
            with torch.no_grad():
                x_rec_input = self.state_history_.detach()
                for i in np.arange(0, Nx, parallel):
                    x_rec_i = self.rec(x[:, :, i:i+parallel], x_rec_input[:, :, i:i+parallel]).detach()
                    x_rec.append(x_rec_i)
                x_rec = torch.cat(x_rec, dim=2)

        return x_rec

    def forward_data_assimilation_ensf(self, data, observations, encoder, noise_level, device, equilibrium=False, latent_state=False, optimal_latent=None,
                                       scaling=500, obs_sigma=0.1, eps_alpha=0.05, euler_steps=100):
        # x has shape (Ni, Nt, Nx, dimx), observations have shape (Nt, Nx, dimx), u has shape (2)
        u = data["u"]
        x = data["x"]
        dt = data["dt"]
        self.ensemble_size = u.shape[0]

        assert len(u.shape) == 2

        observation_vector = observations.reshape([observations.shape[0], -1])
        observation_vector = observation_vector + torch.randn_like(observation_vector) * noise_level
        encoded_obs = encoder(observation_vector.unsqueeze(0)).squeeze(0)
        self.state = torch.zeros(x.shape[0], self.num_latent_states, device=u.device)
        self.state_history, u_history = [], []

        forward_time = 0
        for i in range(x.shape[1]):
            if i == 0 and optimal_latent is not None:
                obs_ti = torch.cat((u, optimal_latent), dim=1)
            else:
                obs_ti = encoded_obs[i, :]
            scaled_obs_ti = obs_ti * scaling
            input = torch.cat((u, self.state), dim=1)

            start_time = time.time()
            with torch.no_grad():
                if equilibrium:
                    self.state = self.state + dt * (self.dyn(input) - self.dyn(torch.zeros_like(input)))
                else:
                    self.state = self.state + dt * self.dyn(input)
            forward_time += time.time() - start_time

            def g_tau(t):
                return 1 - t

            def score_likelihood(xt, t):
                # analytical ∇z log P(Yt+1|z)
                score_x = -(xt - scaled_obs_ti) / obs_sigma**2
                return g_tau(t) * score_x

            latent = torch.concatenate((self.state.view(self.state.shape[0], -1), u), dim=1)
            post_state = self.reverse_SDE(x0=latent * scaling, score_likelihood=score_likelihood,
                                          time_steps=euler_steps, device=device, eps_alpha=eps_alpha) / scaling
            self.state = post_state[:, :self.state.shape[1]]
            u = post_state[:, self.state.shape[1]:]
            self.state_history.append(self.state)
            u_history.append(u)

        self.state_history = torch.stack(self.state_history).transpose(0, 1)
        u_history = torch.stack(u_history).transpose(0, 1)

        if latent_state:
            return self.state_history, u_history, encoded_obs

        Ni = self.state_history.shape[0]
        Nt = self.state_history.shape[1]
        Nx = x.shape[2]
        dims = self.state_history.shape[2]
        self.state_history_ = self.state_history.unsqueeze(2).expand(Ni, Nt, Nx, dims)

        x_rec = []
        with torch.no_grad():
            x_rec_input = self.state_history_.detach()
            for i in np.arange(0, Nx, parallel):
                x_rec_i = self.rec(x[:, :, i:i+parallel], x_rec_input[:, :, i:i+parallel]).detach()
                x_rec.append(x_rec_i)
            x_rec = torch.cat(x_rec, dim=2)

        return x_rec

    def reverse_SDE(self, x0, score_likelihood=None, time_steps=100, save_path=False, device="cpu", eps_alpha=0.05):
        ensemble_size = x0.shape[0]

        def cond_alpha(t):
            return 1 - (1 - eps_alpha) * t

        def cond_sigma_sq(t):
            return t

        def f_func(t):
            # f = d(log_alpha)/dt
            alpha_t = cond_alpha(t)
            return -(1 - eps_alpha) / alpha_t

        def g_sq(t):
            # g = d(sigma_t^2)/dt - 2f * sigma_t^2
            return 1 - 2 * f_func(t) * cond_sigma_sq(t)

        def g_func(t):
            return np.sqrt(g_sq(t))

        dt = 1.0 / time_steps
        xt = torch.randn(ensemble_size, x0.shape[1], device=device)
        t = 1.0

        if save_path:
            path_all = [xt]
            t_vec = [t]

        for i in range(time_steps):
            alpha_t = cond_alpha(t)
            sigma2_t = cond_sigma_sq(t)
            diffuse = g_func(t)

            if score_likelihood is not None:
                xt += -dt * (f_func(t) * xt + diffuse**2 * ((xt - alpha_t * x0) / sigma2_t) - diffuse**2 * score_likelihood(xt, t)) \
                    + np.sqrt(dt) * diffuse * torch.randn_like(xt)
            else:
                xt += -dt * (f_func(t) * xt + diffuse**2 * ((xt - alpha_t * x0) / sigma2_t)) + np.sqrt(dt) * diffuse * torch.randn_like(xt)

            if save_path:
                path_all.append(xt)
                t_vec.append(t)

            t = t - dt

        if save_path:
            return path_all, t_vec
        else:
            return xt
