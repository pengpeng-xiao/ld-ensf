
"""
Script that solves that solves the 2D shallow water equations using finite
differences where the momentum equations are taken to be linear, but the
continuity equation is solved in its nonlinear form. The model supports turning
on/off various terms, but in its mst complete form, the model solves the following
set of eqations:

    du/dt - fv = -g*d(eta)/dx + tau_x/(rho_0*H)- kappa*u
    dv/dt + fu = -g*d(eta)/dy + tau_y/(rho_0*H)- kappa*v
    d(eta)/dt + d((eta + H)*u)/dx + d((eta + H)*u)/dy = sigma - w
    
with boundary conditions u(L,y,t) = v(x,L,t) = 0 at the boundaries.
where f = f_0 + beta*y can be the full latitude varying coriolis parameter.
For the momentum equations, an ordinary forward-in-time centered-in-space
scheme is used. However, the coriolis terms is not so trivial, and thus, one
first finds a predictor for u, v and then a corrected value is computed in
order to include the coriolis terms. In the continuity equation, it's used a
forward difference for the time derivative and an upwind scheme for the non-
linear terms. The model is stable under the CFL condition of

    dt <= min(dx, dy)/sqrt(g*H)    and    alpha << 1 (if coriolis is used)

where dx, dy is the grid spacing in the x- and y-direction respectively, g is
 the acceleration of gravity and H is the resting depth of the fluid.

The saved data should be a dictionary with the following keys:
 y, u, x, dt, (y_x, y_u)
"""

import time
import numpy as np
# import viz_tools
import random
from tqdm import tqdm 
from src.utils import set_seed

set_seed(42)
# set_seed(43)
# set_seed(0)

# ==================================================================================
# ================================ Parameter stuff =================================
# ==================================================================================
# --------------- Physical prameters ---------------
L_x = 1E+6              # Length of domain in x-direction
L_y = 1E+6              # Length of domain in y-direction
g = 9.81                 # Acceleration of gravity [m/s^2]
H = 100                # Depth of fluid [m]
f_0 = 1E-4              # Fixed part ofcoriolis parameter [1/s]
beta = 2E-11            # gradient of coriolis parameter [1/ms]
rho_0 = 1024.0          # Density of fluid [kg/m^3)]
tau_0 = 0.1             # Amplitude of wind stress [kg/ms^2]
use_coriolis = True     # True if you want coriolis force
use_beta = True         # True if you want variation in coriolis

# --------------- Computational prameters ---------------
Ni = 200
Nx = 150                            # Number of grid points in x-direction
Ny = 150                            # Number of grid points in y-direction
Nt = 2000                 # Total number of time steps in simulation
nt = 400                 # Number of saved time steps
dx = L_x/(Nx - 1)                   # Grid spacing in x-direction
dy = L_y/(Ny - 1)                   # Grid spacing in y-direction
dt = 0.1*min(dx, dy)/np.sqrt(g*H)    # Time step (defined from the CFL condition)
x = np.linspace(-L_x/2, L_x/2, Nx)  # Array with x-points
y = np.linspace(-L_y/2, L_y/2, Ny)  # Array with y-points
X, Y = np.meshgrid(x, y)             # Meshgrid for plotting
X = np.transpose(X)                  # To get plots right
Y = np.transpose(Y)                  # To get plots right
# Define friction array if friction is enabled.

# Define coriolis array if coriolis is enabled.
if (use_coriolis is True):
    if (use_beta is True):
        f = f_0 + beta*y        # Varying coriolis parameter
        L_R = np.sqrt(g*H)/f_0  # Rossby deformation radius
        c_R = beta*g*H/f_0**2   # Long Rossby wave speed
    else:
        f = f_0*np.ones(len(y))                 # Constant coriolis parameter

    alpha = dt*f                # Parameter needed for coriolis scheme
    beta_c = alpha**2/4         # Parameter needed for coriolis scheme

inputs = np.zeros((Ni, 2))
outputs = np.zeros((Ni, nt +1, 3, Nx, Ny))
for i in tqdm(range(Ni)): 
    # time_step = 1                        # For counting time loop steps
    # ==================================================================================
    # ==================== Allocating arrays and initial conditions ====================
    # ==================================================================================
    u_n = np.zeros((Nx, Ny))      # To hold u at current time step
    u_np1 = np.zeros((Nx, Ny))    # To hold u at next time step
    v_n = np.zeros((Nx, Ny))      # To hold v at current time step
    v_np1 = np.zeros((Nx, Ny))    # To hold v at enxt time step
    eta_n = np.zeros((Nx, Ny))    # To hold eta at current time step
    eta_np1 = np.zeros((Nx, Ny))  # To hold eta at next time step

    # Temporary variables (each time step) for upwind scheme in eta equation
    h_e = np.zeros((Nx, Ny))
    h_w = np.zeros((Nx, Ny))
    h_n = np.zeros((Nx, Ny))
    h_s = np.zeros((Nx, Ny))
    uhwe = np.zeros((Nx, Ny))
    vhns = np.zeros((Nx, Ny))

    # Initial conditions for u and v.
    u_n[:, :] = 0.0             # Initial condition for u
    v_n[:, :] = 0.0             # Initial condition for u
    u_n[-1, :] = 0.0            # Ensuring initial u satisfy BC
    v_n[:, -1] = 0.0            # Ensuring initial v satisfy BC

    # =============== Done with setting up arrays and initial conditions ===============

    t_0 = time.perf_counter()  # For timing the computation loop

        # ============================= Parameter stuff done ===============================
    # Initial condition for eta.
    # input = np.random.rand(2)
    input = np.random.rand(2) * 0.5
    
    eta_0 = np.exp(-((X+L_x/2-input[0]*L_x)**2/(2*(0.05E+6)**2) + (Y+L_y/2-input[1]*L_y)**2/(2*(0.05E+6)**2)))
    # eta_0_mu1 = 4.0e-10*L_x*(-L_x * input[0] + L_x/2 + x) * \
    #     np.exp(-2.0e-10 * (-L_x * input[0] + L_x/2 + x)**2 - 2.0e-10 * (-L_y * input[1] + L_y/2 + y)**2)
    # eta_0_mu2 = 4.0e-10 * L_y*(-L_y * input[1] + L_y/2 + y) *\
    #      np.exp(-2.0e-10 * (-L_x * input[0] + L_x / 2 + x)**2 - 2.0e-10 * (-L_y * input[1] + L_y/2 + y)**2)

    eta_n = eta_0
    outputs[i, 0, :, :, :] = np.stack([u_n, v_n, eta_n])
    
    for t in range(Nt):
        # ------------ Computing values for u and v at next time step --------------
        u_np1[:-1, :] = u_n[:-1, :] - g*dt/dx*(eta_n[1:, :] - eta_n[:-1, :])
        v_np1[:, :-1] = v_n[:, :-1] - g*dt/dy*(eta_n[:, 1:] - eta_n[:, :-1])
        
                    # Use a corrector method to add coriolis if it's enabled.
        if (use_coriolis is True):
            u_np1[:, :] = (u_np1[:, :] - beta_c*u_n[:, :] + alpha*v_n[:, :])/(1 + beta_c)
            v_np1[:, :] = (v_np1[:, :] - beta_c*v_n[:, :] - alpha*u_n[:, :])/(1 + beta_c)
            
        v_np1[:, -1] = 0.0      # Northern boundary condition
        u_np1[-1, :] = 0.0      # Eastern boundary condition
        # -------------------------- Done with u and v -----------------------------

        # --- Computing arrays needed for the upwind scheme in the eta equation.----
        h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
        h_e[-1, :] = eta_n[-1, :] + H

        h_w[0, :] = eta_n[0, :] + H
        h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H) 
        h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
        h_n[:, -1] = eta_n[:, -1] + H

        h_s[:, 0] = eta_n[:, 0] + H
        h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

        uhwe[0, :] = u_np1[0, :]*h_e[0, :]
        uhwe[1:, :] = u_np1[1:, :]*h_e[1:, :] - u_np1[:-1, :]*h_w[1:, :]

        vhns[:, 0] = v_np1[:, 0]*h_n[:, 0]
        vhns[:, 1:] = v_np1[:, 1:]*h_n[:, 1:] - v_np1[:, :-1]*h_s[:, 1:]
        # ------------------------- Upwind computations done -------------------------
        # ----------------- Computing eta values at next time step -------------------
        eta_np1[:, :] = eta_n[:, :] - dt*(uhwe[:, :]/dx + vhns[:, :]/dy)    # Without source/sink
        # ----------------------------- Done with eta --------------------------------
        
        u_n = np.copy(u_np1)        # Update u for next iteration
        v_n = np.copy(v_np1)        # Update v for next iteration
        eta_n = np.copy(eta_np1)    # Update eta for next iteration

        # time_step += 1  
        
        outputs[i, t// (Nt//nt) +1, :, :, :] = np.stack([u_n, v_n, eta_n])
        # outputs[i, t// (Nt//nt), :, :, :] = np.stack([u_n, v_n, eta_n])
        
        inputs[i] = input
        # if t >= 2000 and (t-2000) % ((Nt-2000)//nt) == 0:
        #     outputs[i, (t-2000)//((Nt-2000)//nt) +1, :, :, :] = np.stack([u_np1, v_np1, eta_np1])
        #     inputs[i] = input

coords = np.stack([X.ravel(),Y.ravel()],axis=1)
outputs = outputs.reshape(Ni, nt+1, 3, Nx, Ny)

np.savez_compressed(f"data/data_2000_150x150_coriolis_ic_nt_400.npz", 
                    u = inputs, x = coords, y = outputs, dt = dt * (Nt//nt), allow_pickle=True)
