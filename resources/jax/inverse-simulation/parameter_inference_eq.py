import functools as ft
from types import SimpleNamespace
from typing import Optional
import os
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
#os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                           "intra_op_parallelism_threads=1")

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.tree_util as jtu
import pandas as pd 
import diffrax as dfx 
import optax 
import matplotlib.pyplot as plt
import time 
import json
import time 
jax.config.update('jax_array', True)

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
class LTISystem(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    C: jnp.ndarray
    D: jnp.ndarray

@eqx.filter_jit
def zone_lti(Cai, Cwe, Cwi, Re, Ri, Rw, Rg):
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

    D = jnp.array([0])

    return LTISystem(A, B, C, D)


def interpolate_us(ts, us, B):
    if us is None:
        m = B.shape[-1]
        u_t = SimpleNamespace(evaluate=lambda t: jnp.zeros((m,)))
    else:
        u_t = dfx.LinearInterpolation(ts=ts, ys=us)
    return u_t

# solve the zone ode
@eqx.filter_jit
def diffeqsolve(
    rhs,
    ts: jnp.ndarray,
    y0: jnp.ndarray,
    solver: dfx.AbstractSolver = dfx.Euler(),#dfx.Dopri5(),
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    dt0: float = 3600.,
) -> jnp.ndarray:
    return dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=ts[0],
        t1=ts[-1],
        y0=y0,
        dt0=dt0,
        saveat=dfx.SaveAt(ts=ts),
    ).ys
    
@eqx.filter_jit
def forward_simulation(
    p: jnp.ndarray,
    x0: jnp.ndarray,
    ts: jnp.ndarray,
    us: Optional[jnp.ndarray] = None,
    std_measurement_noise: float = 0.0,
    key=jr.PRNGKey(1),
    ):
    sys = zone_lti(*p)
    u_t = interpolate_us(ts, us, sys.B)

    def rhs(t, y, args):
        return sys.A @ y + sys.B @ u_t.evaluate(t)

    xs = diffeqsolve(rhs, ts, x0)

    # noisy measurements
    ys = xs @ sys.C.transpose()
    ys = ys + jr.normal(key, shape=ys.shape) * std_measurement_noise
    return xs, ys

class Zone(eqx.Module):
    #sys: LTISystem
    p: jnp.ndarray
    x0: jnp.ndarray
    std_measurement_noise: float 
    key = jr.PRNGKey(1)

    def __call__(self, ts, us: Optional[jnp.ndarray] = None):
        #Cai, Cwe, Cwi, Re, Ri, Rw, Rg = self.p
        sys = zone_lti(*self.p)
        u_t = interpolate_us(ts, us, sys.B)

        def rhs(t, y, args):
            return sys.A @ y + sys.B @ u_t.evaluate(t)

        # solve for ODE
        xs = diffeqsolve(rhs, ts, self.x0)

        # get output
        ys = xs @ sys.C.transpose()
        ys = ys + jr.normal(self.key, shape=ys.shape) * self.std_measurement_noise

        return xs, ys
       
## Get measurement data
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
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

u_train = data_train.values[:,:5]
y_train = data_train.values[:,5]

## Run optimization for inference
# parameter settings
p_lb = jnp.array([1.0E3, 1.0E4, 1.0E5, 1.0, 1E-02, 1.0, 1.0E-1])
p_ub = jnp.array([1.0E5, 1.0E6, 1.0E7, 10., 10., 100., 10.])
x_lb = jnp.array([y_train[0], 20.0, 20.0])
x_ub = jnp.array([y_train[0], 35.0, 30.0])

p0 = p_ub
x0 = x_ub#jnp.array([y_train[0], 30.0, 25.0])
print(p0, x0)

ts = jnp.arange(0, n_train, 1)*dt

# Instantiate zone model
std_measurement_noise = 0.0
zone = Zone(p0, x0, std_measurement_noise)
# test and time
test_ts = time.time()
forward_simulation(zone.p, zone.x0, ts, u_train)
test_te = time.time()
print(f"single forward simulation costs {test_te - test_ts} s!")

# loss function
# gradients should only be able to change Q/R parameters
# *not* the model (well at least not in this example :)
filter_spec = jtu.tree_map(lambda arr: False, zone)
filter_spec = eqx.tree_at(
    lambda tree: (tree.p, tree.x0), filter_spec, replace=(True, True)
)

@eqx.filter_jit
@ft.partial(eqx.filter_value_and_grad, arg=filter_spec)
def loss_fcn(zone, ts, us, ys_true, p_lb, p_ub, x_lb, x_ub):
    xs, ys_pred = zone(ts, us)
    loss = jnp.mean((ys_pred[1:] - ys_true)**2)
    penalty = jnp.sum(jax.nn.relu(zone.p - p_ub) + jax.nn.relu(p_lb - zone.p))+\
              jnp.sum(jax.nn.relu(zone.x0 - x_ub) + jax.nn.relu(x_lb - zone.x0))

    return loss + penalty
    
n_epochs = 10000
print_every = 100
schedule = optax.exponential_decay(
    init_value = 1e-01, 
    transition_steps = 1000, 
    decay_rate = 0.99, 
    transition_begin=0, 
    staircase=False, 
    end_value=1e-06
)
optimizer = optax.chain(
    optax.adabelief(learning_rate = schedule)
)


def fit(data, n_epochs, zone: eqx.Module, optimizer: optax.GradientTransformation, p_lb, p_ub, x_lb, x_ub):
    # initialize params
    opt_state = optimizer.init(zone)
    (ts, us, ys) = data

    @jax.jit
    def step(zone, opt_state, ts, us, ys, p_lb, p_ub, x_lb, x_ub):
        loss, grads = loss_fcn(zone, ts, us, ys, p_lb, p_ub, x_lb, x_ub)
        updates, opt_state = optimizer.update(grads, opt_state)
        zone = eqx.apply_updates(zone, updates)

        return zone, opt_state, loss, grads

    for epoch in range(n_epochs):
        zone, opt_state, loss, grads = step(zone, opt_state, ts, us, ys, p_lb, p_ub, x_lb, x_ub)
        if epoch % 100 == 0:
            print(f'epoch: {epoch}, training loss: {loss}')
            #print(grads['rc'].max(), grads['rc'].min())

    return zone

xy_train = (ts, u_train, y_train)
s = time.time()
zone = fit(xy_train, n_epochs, zone, optimizer, p_lb, p_ub, x_lb, x_ub)
e = time.time()
print(f"execution time is: {e-s} seconds !")

## run for performance check
# forward simulation with infered parameters for the whole data set
xs_pred, ys_pred = forward_simulation(zone.p, zone.x0, ts, u_train)

print(xs_pred.shape, ys_pred.shape)
print(y_train.shape)

plt.figure(figsize=(12,6))
plt.plot(y_train, 'b-', label='Target')
plt.plot(ys_pred, 'r-', label="Prediction")
plt.ylabel('Temperature (C)')
plt.legend()
plt.savefig('parameter_inference.png')

# save the parameters
params_tolist = [float(p) for p in zone.p] + [float(x0) for x0 in zone.x0]
with open('zone_coefficients.json', 'w') as f:
    json.dump(params_tolist,f)
