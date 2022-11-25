import jax
import jax.numpy as jnp
import pandas as pd 
import numpy as np
from jax import jit, lax
from jax import grad
from diffrax import diffeqsolve, ODETerm, Euler, Dopri5, SaveAt, PIDController
import optax 
import matplotlib.pyplot as plt
import time 
import json
from functools import partial 

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
def zone_state_space(t, x, A, B, d):
    x = x.reshape(-1, 1)
    d = d.reshape(-1, 1)
    dx = jnp.matmul(A, x) + jnp.matmul(B, d)
    dx = dx.reshape(-1)

    return dx

# Using for loop to update the disturbance every time step


@partial(jit, static_argnums=(0, 1, 2, 3,))
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

    # initialize output
    #t_all = [t]
    #x_all = [x0]

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


### Parameter Inference

# load training data - 1-min sampling rate
data = pd.read_csv('./data/disturbance_1min.csv', index_col=[0])
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
solver = Euler()

# forward steps
f = lambda t, x, args: zone_state_space(t, x, *args)#args[0], args[1], args[2]) 
def forward_parameters(p, x, ts, te, dt, solver, d):
    """
    p is [Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0]
    x is Tz0
    """
    Cai, Cwe, Cwi, Re, Ri, Rw, Rg, Twe0, Twi0 = p['rc']
    A, B, C, D = get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg)
    args = (A, B, d)

    # intial point
    x0 = jnp.array([x, Twe0, Twi0])

    # forward calculation
    t, x = forward(f, ts, te, dt, x0, solver, args)

    return t, x 

model = lambda p,x: forward_parameters(p, x, ts, te, dt, solver, d)

# loss function
def loss_fcn(p, x, y_true):
    _, y_pred = model(p, x)
    loss = jnp.mean((y_pred[1:,0] - y_true)**2)

    return loss

# data preparation
d = data_train.values[:,:5]
y_train = data_train.values[:,5]

p0 = jnp.array([6953.9422092947289, 21567.368048437285,
              188064.81655062342, 1.4999822619982095, 
              0.55089086571081913, 5.6456475609117183, 
              3.9933826145529263, 32., 26.])

x0 = y_train[0]
print(p0, x0)
print(loss_fcn({'rc':p0}, x0, y_train))

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

def fit(data, n_epochs, params: optax.Params, optimizer: optax.GradientTransformation) -> optax.Params:
    # initialize params
    states = optimizer.init(params)
    x, y = data 

    @jit
    def step(params, states, x, y):
        loss, grads = jax.value_and_grad(loss_fcn)(params, x, y)
        updates, states = optimizer.update(grads, states, params)
        params = optax.apply_updates(params, updates)

        return params, states, loss, grads
    
    for epoch in range(n_epochs):
        params, states, loss, grads = step(params, states, x, y)
        if epoch % 1000 == 0:
            print(f'epoch {epoch}, training loss: {loss}')
            print(grads['rc'].max(), grads['rc'].min())

    return params

n_epochs = 50000
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
params = fit((x0, y_train), n_epochs, initial_params, optimizer)

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
