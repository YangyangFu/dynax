import os
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
#os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                           "intra_op_parallelism_threads=1")
import functools as ft
from functools import partial 
from types import SimpleNamespace
from typing import Optional

import diffrax as dfx
import equinox as eqx  # https://github.com/patrick-kidger/equinox
from flax import linen as nn
import jax
from jax import jit, lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from diffrax import diffeqsolve, ODETerm, Euler, ImplicitEuler, Kvaerno3, Dopri5, SaveAt, PIDController, NewtonNonlinearSolver

import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import time 
import pandas as pd 

# enable 64 bit. This only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
#config.update("jax_debug_nans", True)
#config.update("jax_disable_jit", True)

n_devices = jax.local_device_count()
print(n_devices)
# a simple RC zone model -> ODE system
"""
This is a 4r3c model for zone temperature prediction
ODE::   xdot = Ax + Bd
        y = Cx
States:
        x = [Tai, Twe, Twi]'
Disturbances:
        u = [Tao, qCon_i, qHVAC, qRad_e, qRad_i]'
Output:
        y = [Tai]
===================================================
"""
@jax.jit
def get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg):
    A = jnp.zeros((3, 3))
    B = jnp.zeros((3, 5))
    C = jnp.zeros((1, 3))
    A = A.at[0, 0].set(-1/Cai*(1/Rg+1/Ri))
    A = A.at[0, 2].set(1/(Cai*Ri))
    A = A.at[1, 1].set(-1/Cwe*(1/Re+1/Rw))
    A = A.at[1, 2].set(1/(Cwe*Rw))
    A = A.at[2, 0].set(1/(Cwi*Ri))
    A = A.at[2, 1].set(1/(Cwi*Rw))
    A = A.at[2, 2].set(-1/Cwi*(1/Rw+1/Ri))

    B = B.at[0, 0].set(1/(Cai*Rg))
    B = B.at[0, 1].set(1/Cai)
    B = B.at[0, 2].set(1/Cai)
    B = B.at[1, 0].set(1/(Cwe*Re))
    B = B.at[1, 3].set(1/Cwe)
    B = B.at[2, 4].set(1/Cwi)

    C = C.at[0, 0].set(1)

    D = 0

    return A, B, C, D

@jit
def continuous_kmf(t, xP, A, B, C, Q, R, u, z):
    # extract states
    x, P = xP
    
    # eq 3.22 of Ref [1]
    K = P @ C.transpose() @ jnp.linalg.inv(R)

    # eq 3.21 of Ref [1]
    dPdt = (
        A @ P
        + P @ A.transpose()
        + Q
        - P @ C.transpose() @ jnp.linalg.inv(R) @ C @ P
    )

    # eq 3.23 of Ref [1]
    dxdt = A @ x + B @ u + K @ (z - C @ x)
    #jax.debug.print("{x}", x=x)
    #jax.debug.print("{dxdt}", dxdt=A@x + B@u)
    #jax.debug.print("{K}", K=K)
    #jax.debug.print("{e}", e=z - C@x)
    #jax.debug.print("{dxdt}", dxdt=dxdt)
    return (dxdt, dPdt)

# Using for loop to update the disturbance every time step
# forward function that simulates ODE given time period and inputs
@partial(jit, static_argnums=(0, 1, 2, 3, 5,))
def forward(func, ts, te, dt, xP0, solver, args):
    # unpack args:
    # A, B, C: system dynamics coefficents
    # Q, R: variance of modeling errors and measurement noise
    # u: control inputs and disturbances
    # z: measurements
    A, B, C, Q, R, u, z = args

    # ode formulation
    term = ODETerm(func)

    # initial step
    t = ts
    tnext = t + dt
    u0 = u[0, :]
    z0 = z[0, :]
    args = (A, B, C, Q, R, u0, z0)
    
    # helper function
    def step_at_t(carryover, t, term, dt, te, A, B, C, Q, R, u, z):
        # the lax.scannable function to computer ODE/DAE systems
        xP, state, i = carryover
        args = (A, B, C, Q, R, u[i, :], z[i, :])
        tnext = jnp.minimum(t + dt, te)

        xPnext, _, _, state, _ = solver.step(
            term, t, tnext, xP, args, state, made_jump=False)
        i += 1

        return (xPnext, state, i), xP
    
    # main loop
    state0 = solver.init(term, t, tnext, xP0, args)
    i = 0
    carryover_init = (xP0, state0, i)
    step_func = partial(step_at_t, term=term, dt=dt, te=te, A=A, B=B, C=C, Q=Q, R=R, u=u, z=z)
    time_steps = jnp.arange(ts, te+1, dt)
    carryover_final, xP_all = lax.scan(
        step_func, init=carryover_init, xs=time_steps)

    return time_steps, xP_all

"""
### A single forward pass test
### ==============================================
# load training data - 1-min sampling rate
data = pd.read_csv('./data/disturbance_1min.csv', index_col=[0])
index = range(0, len(data)*60, 60)
data.index = index

# sample every hour
dt = 3600.
data = data.groupby([data.index // dt]).mean()
n = len(data)

# split training and testing
ratio = 0.1
n_train = int(len(data)*ratio)
print(n_train)
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

us_train = jnp.array(data_train.iloc[:n_train+1,:5].values)
#us = None 

# define training parameters 
ys_train = jnp.array(data_train.iloc[:, -1].values).reshape(-1,1)

# test forward function
kmf = lambda t, xP, args: continuous_kmf(t, xP, *args)
ts = 0
te = n_train*3600 #14*24*3600
x0 = jnp.array([20.0, 35.0, 26.0])
# weighs how much we trust our initial guess: if we know the exact position and velocity, we give it a zero covariance matrix
P0 = jnp.diag(jnp.ones((3,)))*0.1
xP0 = (x0, P0)
solver = Euler()
RC = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394,
                     1.1442776918411255, 5.741048812866211])
A, B, C, D = get_ABCD(*RC)
# weighs how much we trust our model of the system
Q = jnp.diag(jnp.ones((3,)))*1e-05
# weighs how much we trust in the measurements of the system
R = jnp.diag(jnp.ones((1,)))*10000
args = (A, B, C, Q, R, us_train, ys_train)
forward_ts = time.time()
t, xP = forward(kmf, ts, te, dt, xP0, solver, args)
forward_te = time.time()
print(f"single forward simulation costs {forward_te-forward_ts} s!")
print(t.shape)
print(xP)
"""

# load training data - 1-min sampling rate
data = pd.read_csv('./data/disturbance_1min.csv', index_col=[0])
index = range(0, len(data)*60, 60)
data.index = index

# sample 
dt = 900.
data = data.groupby([data.index // dt]).mean()
n = len(data)

# split training and testing
ratio = 0.75
n_train = int(len(data)*ratio)
print(n_train)
print(data.head(5))
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

us_train = jnp.array(data_train.iloc[:n_train+1,:5].values)
#us = None 

# define training parameters 
ys_train = jnp.array(data_train.iloc[:, -1].values).reshape(-1,1)

# test forward function
kmf = lambda t, xP, args: continuous_kmf(t, xP, *args)
ts = 0
te = n_train*900 #14*24*3600
nonl_solver = NewtonNonlinearSolver(rtol=1e-3, atol=1e-6)
solver = ImplicitEuler(nonlinear_solver=nonl_solver) #Euler()
RC = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394,
                     1.1442776918411255, 5.741048812866211])
A, B, C, D = get_ABCD(*RC)

# some initials
x0 = jnp.array([20.0, 20.0, 20.0])
P0 = jnp.diag(jnp.ones((3,)))
xP0 = (x0, P0)

# weighs how much we trust our model of the system
Q0 = jnp.diag(jnp.ones((3,)))*1e-01
# weighs how much we trust in the measurements of the system
R0 = jnp.diag(jnp.ones((1,)))*10000

# real measurement with noise
std_measurement_noise = 0.
key = jr.PRNGKey(1,)
ys_train_noise = ys_train + \
    jr.normal(key, shape=ys_train.shape) * std_measurement_noise

# some functions
### parameter inference
### ====================
# create function to faciliate the simulation of kalman filter with different Q and R
def forward_parameters(p, xP0, ts, te, dt, solver, args):
    """
    p is [Q, R]
    x is Tz0
    """
    Q, R = p['qr']
    A, B, C, u, z = args
    args1 = (A, B, C, Q, R, u, z)

    # forward calculation
    t, xP = forward(kmf, ts, te, dt, xP0, solver, args1)
    
    #jax.debug.print("{xP}", xP=xP)
    return t, xP

args = (A, B, C, us_train, ys_train_noise)
def model(p, xP0): return forward_parameters(p, xP0, ts, te, dt, solver, args)

model = jit(model)

@jit
def loss_fn(params, xP0, y_true):
    _, xP_pred = model(params, xP0) # y_meas is used in this model to construct y_esti
    y_esti = xP_pred[0]  # get zone temperature state
    P_esti = xP_pred[1]
    loss = jnp.mean((y_esti[1:, 0] - y_true)**2)

    penality = 0  # jnp.linalg.norm(P_pred)
    return loss + penality


def fit(data, n_epochs, params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    # initialize params
    states = optimizer.init(params)
    x, y = data

    @jit
    def step(params, states, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, states = optimizer.update(grads, states, params)
        params = optax.apply_updates(params, updates)

        return params, states, loss, grads

    for epoch in range(n_epochs):
        params, states, loss, grads = step(params, states, x, y)
        if epoch % 1 == 0:
            print(f'epoch {epoch}, training loss: {loss}')
            #print(grads['rc'].max(), grads['rc'].min())

    return params


# test loss function
print(loss_fn({'qr':(Q0, R0)}, xP0, ys_train))

# trainer
n_epochs = 1200
schedule = optax.exponential_decay(
    init_value = 1e-5, 
    transition_steps = 500, 
    decay_rate = 0.99, 
    transition_begin=0, 
    staircase=False, 
    end_value=1e-8
)
optimizer = optax.chain(
    optax.adabelief(learning_rate = schedule)
)

initial_params = {'qr': (Q0, R0)}
s = time.time()
params = fit((xP0, ys_train_noise), n_epochs, initial_params, optimizer)
e = time.time()
print(f"execution time is: {e-s} seconds !")
print(f"Final Q: \n{params['qr'][0]}\n Final R: \n{params['qr'][1]}")

## PLOTS

t, xPhat = model(params, xP0)
plt.figure(figsize=(12,6))
plt.plot(t[1:], ys_train, label="true", color="orange")
plt.plot(
    t,
    xPhat[0][:, 0],
    label="estimated",
    color="blue",
    linestyle="dashed",
)
plt.plot(
    t[1:],
    ys_train_noise,
    label="measured",
    color='red',
    linestyle='dashed',
    linewidth=0.5,
)

plt.xlabel("time")
plt.ylabel("Temperature")
plt.grid()
plt.legend()
plt.title("Kalman-Filter optimization w.r.t Q/R")
plt.savefig('kalman_filter.png')
plt.savefig('kalman_filter.pdf')
