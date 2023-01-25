import os
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
#os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                           "intra_op_parallelism_threads=1")

import jax
import jax.numpy as jnp
import tensorflow as tf
import pandas as pd 
import numpy as np
from jax import jit, lax, vmap
from jax import grad
from diffrax import diffeqsolve, ODETerm, Euler, Dopri5, SaveAt, PIDController
import optax 
import matplotlib.pyplot as plt
import time 
import json
from functools import partial 
import time 

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

@jit
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

    return (dxdt, dPdt)

# Using for loop to update the disturbance every time step
# forward function that simulates ODE given time period and inputs with only one sample of data so that we can parallize using vmap
# simulate with 
@partial(jit, static_argnums=(0, 1, 2, 3,))
def forward(kmf, zone, ts, te, dt, solver, args):
    """
    1. use kalman filter to estimate initial states
    2. calculate future system dynamics based on system model instead of kalman filter dynamics
    """
    
    # unpack args:
    # A, B, C: system dynamics coefficents
    # Q, R: variance of modeling errors and measurement noise
    # u: control inputs and disturbances
    # z: measurements
    A, B, C, Q, R, x = args

    
    # ode formulation
    term = ODETerm(func)

    # initial step
    t = ts
    tnext = t + dt
    u0 = u[0, :]
    args = (A, B, C, Q, R, u0)

    # helper function
    def step_at_t(carryover, t, term, dt, te, A, B, C, Q, R, u):
        # the lax.scannable function to computer ODE/DAE systems
        xP, state, i = carryover
        args = (A, B, C, Q, R, u[i, :])
        tnext = jnp.minimum(t + dt, te)

        xPnext, _, _, state, _ = solver.step(
            term, t, tnext, xP, args, state, made_jump=False)
        i += 1

        return (xPnext, state, i), xP

    # main loop
    state0 = solver.init(term, t, tnext, xP0, args)
    i = 0
    carryover_init = (xP0, state0, i)
    step_func = partial(step_at_t, term=term, dt=dt, te=te,
                        A=A, B=B, C=C, Q=Q, R=R, u=u, z=z)
    time_steps = jnp.arange(ts, te+1, dt)
    carryover_final, xP_all = lax.scan(
        step_func, init=carryover_init, xs=time_steps)

    return time_steps, xP_all

### Parameter Inference
# load training data - 1-min sampling rate
data = pd.read_csv('./data/disturbance_1min.csv', index_col=[0])
index = range(0, len(data)*60, 60)
data.index = index

# sample every hour
dt = 3600
data = data.groupby([data.index // dt]).mean()
n = len(data)

# prepare data for multi-step errors minimization
# for example, we can reduce the 24-step-ahead errors.
# For this purpose, we need prepare a dataset that has a size of (Nsample, 1+N+Nd) 
# Current dataset: 0-4th columns are disturbances to RC zone model, 5th column is the target -> zone temperature
# Three type of data
# - for state x:
#   construct an array of (B, Ns)
# - for disturbance d
#   construct an array of (B, Nq, Nd)
# - for prediction targets y - future temperature
#   construct an array of (B, Nq) 
# TODO: implement pytorch-like data loader to get batched data -> looks like a library to be developed for dynamic systems

N_q = 24 # future steps for predictions
N_s = 3 # number of states
N_d = 5 # number of disturbances
target = data.iloc[:,-1].values
dist = data.iloc[:,:N_d].values # first N_d columns

SEQUENCE_LENGTH = n - (N_q+1)
target_batched = []
dist_batched = []
for i in range(SEQUENCE_LENGTH):
    # get (x, d, y) for each step
    target_i = target[i:i+N_q+1]
    dist_i = dist[i:i+N_q, :]
    # append for batched data
    target_batched.append(target_i) # (B, 1+Nq)
    dist_batched.append(dist_i) # (B, (Nq, Nd)

target_batch = np.array(target_batched)
dist_batch = np.array(dist_batched)
x_init_batch = target_batch[:,0]
y_batch = target_batch[:,1:]

print(target_batch.shape)
print(dist_batch.shape)
print(x_init_batch.shape)
print(y_batch.shape)

# prepare batched x:
# 1. flatten disturbance [N, Nq, Nd] to [N, Nq*Nd]
# 2. combine inital x and flattened disturbance to make a set of x
# 3. load data (x,y) using tensor flow data loader and batch
def dataloader(data):
    x_init, dist, y = data
    N = dist.shape[0] 
    dist = dist.reshape(N,-1)
    x_init = x_init.reshape(-1,1)
    x = np.hstack([x_init, dist])
    
    return tf.data.Dataset.from_tensor_slices((x, y))

def split_data(data, train_ratio=0.7):
    x_init, dist, y = data 
    n = len(x_init)
    n_train = int(n*train_ratio)
    return (x_init[:n_train, ], dist[:n_train, :, :], y[:n_train, :]), (x_init[n_train:, ], dist[n_train:, :, :], y[n_train:, :])

# split training and testing
ratio = 0.75
#n_train = int(len(data)*ratio)
#print(n_train)
#data_train = data[:n_train, :]
#data_test = data[n_train:, :]
data_train, data_test = split_data((x_init_batch, dist_batch, y_batch), ratio)

# batch the data
BATCHSIZE = 4 
SHUFFLE_BUFFER_SIZE = 128
data_train = dataloader(data_train)
data_test = dataloader(data_test)

data_train = data_train.batch(BATCHSIZE)

# examine the data in the first batch
ds = data_train.take(1)
for e in ds:
    print(e)

iter = data_train.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    print(sess.run(el)) 

# define training parameters 
ts = 0
te = ts + N_q*dt
solver = Euler()
RC = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394,
                1.1442776918411255, 5.741048812866211])
A, B, C, D = get_ABCD(*RC)

## The following functions are defined without batch information
## We can use vmap to apply batch calculation to these functions
# forward steps
kmf = lambda t, xP, args: continuous_kmf(t, xP, *args)

def forward_parameters(p, xP0, ts, te, dt, solver, args):
    """
    p is [Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0]
    """

    Cai, Cwe, Cwi, Re, Ri, Rw, Rg = p['rc']
    Q, R, dataset = args
    A, B, C, D = get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg)
    
    args1 = (A, B, C, Q, R, u, z)

    # forward calculation
    t, xP = forward(kmf, ts, te, dt, xP0, solver, args1)

    return t, xP

args = (A, B, C, us_train, ys_train_noise)
def model(p, xP0): return forward_parameters(p, xP0, ts, te, dt, solver, args)

# loss function
def loss_fcn(p, x, y_true, p_lb, p_ub):
    _, y_pred = model(p, x)
    loss = jnp.mean((y_pred[1:,0] - y_true)**2)

    penalty = jnp.sum(jax.nn.relu(p['rc'] - p_ub) + jax.nn.relu(p_lb - p['rc']))

    return loss + penalty

def fit(data, n_epochs, params: optax.Params, optimizer: optax.GradientTransformation, p_lb, p_ub) -> optax.Params:
    # initialize params
    states = optimizer.init(params)
    x, y = data 

    @jit
    def step(params, states, x, y, p_lb, p_ub):
        loss, grads = jax.value_and_grad(loss_fcn)(params, x, y, p_lb, p_ub)
        updates, states = optimizer.update(grads, states, params)
        params = optax.apply_updates(params, updates)

        return params, states, loss, grads
    
    for epoch in range(n_epochs):
        params, states, loss, grads = step(params, states, x, y, p_lb, p_ub)
        if epoch % 1000 == 0:
            print(f'epoch {epoch}, training loss: {loss}')
            #print(grads['rc'].max(), grads['rc'].min())

    return params

## Run optimization for inference
# parameter settings
p_lb = jnp.array([1.0E3, 1.0E4, 1.0E5, 1.0, 1E-02, 1.0, 1.0E-1, 20.0, 20.0])
p_ub = jnp.array([1.0E5, 1.0E6, 1.0E7, 10., 10., 100., 10., 35.0, 30.0])

#p0 = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394, 1.1442776918411255, 5.741048812866211, 34.82638931274414, 26.184139251708984])
# play a bit to check the model
x_init_train, dist_train, y_train = data_train
x_init_test, dist_test, y_test = data_test

# single sample
p0 = p_ub
x0 = x_init_train[0]
d0 = dist_train[0, :, :]
y0 = y_train[0,:]
print(x0.shape, d0.shape, y0.shape)
print(p0, x0)
print(model({'rc':p0}, x0))
print(loss_fcn({'rc':p0}, x0, y_train[:N_q], p_lb, p_ub))

# batch sample
x0 = y_train[0:2].reshape(2, 1)
print(x0.shape)
print(vmap(model, in_axes=(None, 0))({'rc': p0}, x0))

# start to train
n_epochs = 100000
schedule = optax.exponential_decay(
    init_value = 0.1, 
    transition_steps = 1000, 
    decay_rate = 0.99, 
    transition_begin=0, 
    staircase=False, 
    end_value=1e-05
)
optimizer = optax.chain(
    optax.adabelief(learning_rate = schedule)
)

initial_params = {'rc': p0}
s = time.time()
params = fit((x0, y_train), n_epochs, initial_params, optimizer, p_lb, p_ub)
e = time.time()
print(f"execution time is: {e-s} seconds !")

## run for performance check
# forward simulation with infered parameters for the whole data set
ts = 0
te = ts + n*dt
d = data.values[:, :5]
y = data.values[:, 5]
model = lambda p,x: forward_parameters(p, x, ts, te, dt, solver, d)
t_pred, ys_pred = model(params, x0)
y_pred = ys_pred[:,0]

print(t_pred.shape, y_pred.shape)
print(y.shape)

plt.figure(figsize=(12,6))
plt.plot(y, 'b-', label='Target')
plt.plot(y_pred, 'r-', label="Prediction")
plt.ylabel('Temperature (C)')
plt.legend()
plt.savefig('parameter_inference.png')

# save the parameters
params_tolist = [float(p) for p in params['rc']]
with open('zone_coefficients.json', 'w') as f:
    json.dump(params_tolist,f)
