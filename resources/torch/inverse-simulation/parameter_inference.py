import torch
from torch.autograd import grad, backward
import numpy as np
import json
import pandas as pd
from torchdiffeq import odeint 


def get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg):
    A = np.zeros((3, 3))
    B = np.zeros((3, 5))
    C = np.zeros((1, 3))
    A[0, 0] = -1/Cai*(1/Rg+1/Ri)
    A[0, 2] = 1/(Cai*Ri)
    A[1, 1] = -1/Cwe*(1/Re+1/Rw)
    A[1, 2] = 1/(Cwe*Rw)
    A[2, 0] = 1/(Cwi*Ri)
    A[2, 1] = 1/(Cwi*Rw)
    A[2, 2] = -1/Cwi*(1/Rw+1/Ri)

    B[0, 0] = 1/(Cai*Rg)
    B[0, 1] = 1/Cai
    B[0, 2] = 1/Cai
    B[1, 0] = 1/(Cwe*Re)
    B[1, 3] = 1/Cwe
    B[2, 4] = 1/Cwi

    C[0, 0] = 1


    D = 0

    return A, B, C, D


# right-hand side 
def zone_state_space(t, x, A, B, d):
    x = x.reshape(-1, 1)
    d = d.reshape(-1, 1)
    dx = np.matmul(A, x) + np.matmul(B, d)
    dx = dx.reshape(-1)

    return dx
"""
def forward(func, ts, te, dt, x0, solver, args):
    # unpack args
    A, B, d = args

    # ode formulation
    term = ODETerm(func)

    # initial step
    t = ts
    tnext = t + dt
    dprev = d[0, :]
    args = (A, B, dprev)
    state = solver.init(term, t, tnext, x0, args)

    # main loop
    i = 0
    #x = x0

    # jit-ed scan to replace while loop
#    cond_func = lambda t: t < te
    def step_at_t(carryover, t, term, dt, te, A, B, d):
        # the lax.scannable function to computer ODE/DAE systems
        x = carryover[0]
        state = carryover[1]
        i = carryover[2]
        args = (A, B, d[i, :])
        tnext = jnp.minimum(t + dt, te)

        xnext, _, _, state, _ = solver.step(
            term, t, tnext, x, args, state, made_jump=False)
        i += 1

        return (xnext, state, i), x

    carryover_init = (x0, state, i)
    step_func = partial(step_at_t, term=term, dt=dt, te=te, A=A, B=B, d=d)
    time_steps = np.arange(ts, te+1, dt)
    carryover_final, x_all = lax.scan(
        step_func, init=carryover_init, xs=time_steps)

    return time_steps, x_all
"""

def forward(func, ts, te, dt, x0, solver, args):
    # unpack args
    A, B, d = args

    # solve odes
    t = np.arange(ts, te+1, dt)
    x = odeint(func, x0, t, method=solver, args=(A, B, d))

    return t, x


### Parameter Inference

# load training data - 1-min sampling rate
data = pd.read_csv('./disturbance_1min.csv', index_col=[0])
index = range(0, len(data)*60, 60)
data.index = index

# sample every hour
dt = 3600
data = data.groupby([data.index // dt]).mean()
n = len(data)

# split training and testing
ratio = 0.75
n_train = int(len(data)*ratio)
print(n_train)
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

# define training parameters
ts = 0
te = ts + n_train*dt
solver = 'euler'

# scale parameters
scale = np.array([3.0E4, 5.0E5, 5.0E6, 10., 5., 10., 50., 36.0, 30.0])
#scale = jnp.array([1.0E5, 3.0E4, 5.0E5, 5., 5., 5., 5., 36.0, 30.0])
# forward steps

def f(t, x, args): return zone_state_space(
    t, x, *args)  # args[0], args[1], args[2])


def forward_parameters(p, x, ts, te, dt, solver, d, scale):
    """
    p is [Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0]
    x is Tz0
    """
    #Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0 = p['rc']
    rc_norm = p['rc']
    rc = rc_norm*scale
    # scale up to normal range
    Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0 = rc

    A, B, C, D = get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg)
    args = (A, B, d)

    # intial point
    x0 = jnp.array([x, Twe0, Twi0])

    # forward calculation
    t, x = forward(f, ts, te, dt, x0, solver, args)

    return t, x


def model(p, x): return forward_parameters(p, x, ts, te, dt, solver, d, scale)


# loss function
def loss_fcn(p, x, y_true, p_lb, p_ub):
    _, y_pred = model(p, x)
    loss = np.mean((y_pred[1:, 0] - y_true)**2)

    penalty = np.sum(jax.nn.relu(
        p['rc'] - p_ub) + jax.nn.relu(p_lb - p['rc']))

    norm = (p['rc'] - p_lb) / (p_ub - p_lb)
    reg = jnp.linalg.norm(norm, 2)
    return loss + penalty


# data preparation
d = data_train.values[:, :5]
y_train = data_train.values[:, 5]
d = jax.device_put(d)
y_train = jax.device_put(y_train)

"""
# A naive gradient descent implementation
# update 
@jit 
def update(p, x, y_true, lr = 0.1):
    return p - lr*jax.grad(loss_fcn)(p, x, y_true)

for i in range(nepochs):
    p = update(p, x0, y_train, lr)

    if i % 100 == 0:
        print(f"======= epoch: {i}")
        loss = loss_fcn(p, x0, y_train)
        print(f" training loss is: {loss}")

print(p)
print(loss_fcn(p, x0, y_train))
"""


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
#p_lb = jnp.array([1.0E3, 1.0E4, 1.0E5, 1.0, 1E-01, 1.0, 1.0, 20.0, 20.0])
#p_ub = jnp.array([1.0E5, 3.0E4, 3.0E5, 10., 1., 10., 10., 35.0, 30.0])
p_lb = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.65])
p_ub = jnp.array([1., 1., 1., 1., 1., 1., 1., 1., 1.])
#p0 = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394, 1.1442776918411255, 5.741048812866211, 34.82638931274414, 26.184139251708984])

p0 = p_ub
p0 = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8])

x0 = jax.device_put(y_train[0])
print(p0, x0)
print(loss_fcn({'rc': p0}, x0, y_train, p_lb, p_ub))

n_epochs = 100000  # 5000000
schedule = optax.exponential_decay(
    init_value=1e-4,
    transition_steps=150000,
    decay_rate=0.99,
    transition_begin=0,
    staircase=False,
    end_value=1e-05
)
optimizer = optax.chain(
    optax.adamw(learning_rate=1e-04)
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
forward_ts = time.time()
def model(p, x): return forward_parameters(p, x, ts, te, dt, solver, d, scale)


t_pred, ys_pred = model(params, x0)
forward_te = time.time()
print(f"single forward simulation costs {forward_te-forward_ts} s!")
y_pred = ys_pred[:, 0]

print(t_pred.shape, y_pred.shape)
print(y.shape)

plt.figure(figsize=(12, 6))
plt.plot(y, 'b-', label='Target')
plt.plot(y_pred, 'r-', label="Prediction")
plt.ylabel('Temperature (C)')
plt.legend()
plt.savefig('parameter_inference.png')

# save the parameters
params_tolist = [float(p) for p in params['rc']*scale]
with open('zone_coefficients.json', 'w') as f:
    json.dump(params_tolist, f)

A, B, C, D = get_ABCD(*initial_params['rc'][:-2])
print(jnp.linalg.eig(A))

A, B, C, D = get_ABCD(*params['rc'][:-2])
print(jnp.linalg.eig(A))

