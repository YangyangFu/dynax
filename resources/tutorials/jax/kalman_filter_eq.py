import os
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
#os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
#                           "intra_op_parallelism_threads=1")
import functools as ft
from types import SimpleNamespace
from typing import Optional

import diffrax as dfx
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
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

class LTISystem(eqx.Module):
    A: jnp.array
    B: jnp.array
    C: jnp.array

def zone_temperature(A,B,C,D) -> LTISystem:
    return LTISystem(A,B,C)

def interpolate_us(ts, us, B):
    if us is None:
        m = B.shape[-1]
        u_t = SimpleNamespace(evaluate=lambda t: jnp.zeros((m,)))
    else:
        u_t = dfx.LinearInterpolation(ts=ts, ys=us)
    return u_t


def diffeqsolve(
    rhs,
    ts: jnp.ndarray,
    y0: jnp.ndarray,
    solver: dfx.AbstractSolver = dfx.Dopri5(),
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(compile_steps=False),
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

def simulate_lti_system(
    sys: LTISystem,
    y0: jnp.ndarray,
    ts: jnp.ndarray,
    us: Optional[jnp.ndarray] = None,
    std_measurement_noise: float = 0.0,
    dt0: float = 3600.,
    key=jr.PRNGKey(
        1,
    ),
):
    u_t = interpolate_us(ts, us, sys.B)

    def rhs(t, y, args):
        return sys.A @ y + sys.B @ u_t.evaluate(t)

    xs = diffeqsolve(rhs, ts, y0, dt0=dt0)
    # noisy measurements
    ys = xs @ sys.C.transpose()
    ys = ys + jr.normal(key, shape=ys.shape) * std_measurement_noise
    return xs, ys

class KalmanFilter(eqx.Module):
    """Continuous-time Kalman Filter

    Ref:
        [1] Optimal and robust estimation. 2nd edition. Page 154.
        https://lewisgroup.uta.edu/ee5322/lectures/CTKalmanFilterNew.pdf
    """

    sys: LTISystem
    x0: jnp.ndarray
    P0: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray

    def __call__(self, ts, ys, us: Optional[jnp.ndarray] = None):

        A, B, C = self.sys.A, self.sys.B, self.sys.C

        y_t = dfx.LinearInterpolation(ts=ts, ys=ys)
        u_t = interpolate_us(ts, us, B)

        y0 = (self.x0, self.P0)
        
        def rhs(t, y, args):
            x, P = y

            # eq 3.22 of Ref [1]
            K = P @ C.transpose() @ jnp.linalg.inv(self.R)

            # eq 3.21 of Ref [1]
            dPdt = (
                A @ P
                + P @ A.transpose()
                + self.Q
                - P @ C.transpose() @ jnp.linalg.inv(self.R) @ C @ P
            )

            # eq 3.23 of Ref [1]
            dxdt = A @ x + B @ u_t.evaluate(t) + K @ (y_t.evaluate(t) - C @ x)

            return (dxdt, dPdt)

        return diffeqsolve(rhs, ts, y0)[0]

### Parameter Inference
# load training data - 1-min sampling rate
data = pd.read_csv('./data/disturbance_1min.csv', index_col=[0])
index = range(0, len(data)*60, 60)
data.index = index

# sample every hour
dt = 3600
data = data.groupby([data.index // dt]).mean()
n = len(data)
print(n)

# split training and testing
ratio = 0.04
n_train = int(len(data)*ratio)
print(n_train)
data_train = data.iloc[:n_train, :]
data_test = data.iloc[n_train:, :]

us = jnp.array(data_train.iloc[:n_train+1,:5].values)
#us = None 

# define training parameters 
Tz_0 = data_train.iloc[0,-1]
ts = jnp.arange(0.0, n_train*3600, 3600.)
print(ts.shape)
print(us.shape)

p0 = jnp.array([9998.0869140625, 99998.0859375, 999999.5625, 9.94130802154541, 0.6232420802116394,
               1.1442776918411255, 5.741048812866211])
A, B, C, D = get_ABCD(*p0)
sys_true = zone_temperature(A,B,C,D)
sys_true_x0 = jnp.array([Tz_0, 34.82638931274414, 26.184139251708984])
sys_true_std_measurement_noise = 0.5

xs, ys = simulate_lti_system(
    sys_true, sys_true_x0, ts, us, std_measurement_noise=sys_true_std_measurement_noise, dt0 = dt
)

# system model -> here we use perfect model. It should not be perfect in practice
sys_model = sys_true 
# initial state guss -> its not perfect 
sys_model_x0 = jnp.array([20.0, 24.0, 22.0])

## TODO: Why large P0 leads to NANs in the first step?
# seems big Q leads to NANs
# weighs how much we trust our model of the system
Q=jnp.diag(jnp.ones((3,))) * 1E-05
# weighs how much we trust in the measurements of the system
R=jnp.diag(jnp.ones((1,)))*1000
# weighs how much we trust our initial guess: if we know the exact position and velocity, we give it a zero covariance matrix
P0=jnp.diag(jnp.ones((3,))) * 0.1
print(P0)
kmf = KalmanFilter(sys_model, sys_model_x0, P0, Q, R)
print(f"Initial Q: \n{kmf.Q}\n Initial R: \n{kmf.R}")
#print(ts.shape, ys.shape, us.shape)
#dxdt = kmf(ts, ys, us)
#print(dxdt)

# gradients should only be able to change Q/R parameters
# *not* the model (well at least not in this example :)
filter_spec = jtu.tree_map(lambda arr: False, kmf)
filter_spec = eqx.tree_at(
    lambda tree: (tree.Q, tree.R), filter_spec, replace=(True, True)
)

@eqx.filter_jit
@ft.partial(eqx.filter_value_and_grad, arg=filter_spec)
def loss_fn(kmf, ts, ys, us, xs):
    xhats = kmf(ts, ys, us)
    return jnp.mean((xs - xhats) ** 2)

n_gradient_steps = 50000
print_every = 100 

schedule = optax.exponential_decay(
    init_value = 1e-5, 
    transition_steps = 5000, 
    decay_rate = 0.99, 
    transition_begin=0, 
    staircase=False, 
    end_value=1e-12
)

opt = optax.chain(
    optax.adabelief(learning_rate = schedule)
)
opt_state = opt.init(kmf)

for step in range(n_gradient_steps):
    value, grads = loss_fn(kmf, ts, ys, us, xs)
    if step % print_every == 0:
        print(f"Current MSE at step {step} is {value} !!")
    updates, opt_state = opt.update(grads, opt_state, kmf)
    kmf = eqx.apply_updates(kmf, updates)

print(f"Final Q: \n{kmf.Q}\n Final R: \n{kmf.R}")

## PLOTS

xhats = kmf(ts, ys, us)
plt.plot(ts, xs[:, 0], label="true zone", color="orange")
plt.plot(
    ts,
    xhats[:, 0],
    label="estimated zone temp",
    color="orange",
    linestyle="dashed",
)
plt.plot(ts, xs[:, 1], label="true wall out", color="blue")
plt.plot(
    ts,
    xhats[:, 1],
    label="estimated wall out",
    color="blue",
    linestyle="dashed",
)
plt.plot(ts, xs[:, 2], label="true wall in", color="red")
plt.plot(
    ts,
    xhats[:, 2],
    label="estimated wall in",
    color="red",
    linestyle="dashed",
)

plt.xlabel("time")
plt.ylabel("Temperature")
plt.grid()
plt.legend()
plt.title("Kalman-Filter optimization w.r.t Q/R")
plt.savefig('kalman_filter.png')
