import jax
import jax.numpy as jnp
import pandas as pd 
import numpy as np
from jax import jit
from jax import grad
from diffrax import diffeqsolve, ODETerm, Euler, Dopri5, SaveAt, PIDController
import matplotlib.pyplot as plt
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
def zone_state_space(t, x, A, B, d):
    x = x.reshape(-1, 1)
    d = d.reshape(-1, 1)
    dx = jnp.matmul(A, x) + jnp.matmul(B, d)
    dx = dx.reshape(-1)

    return dx


# get some data
dt = 3600
disturbances = pd.read_csv('./disturbance_1min.csv', index_col=[0])
n = len(disturbances)
index = range(0, n*60, 60)
disturbances.index = index
disturbances_dt = disturbances.groupby([disturbances.index // dt]).mean()

Cz = 6953.9422092947289
Cwe = 21567.368048437285
Cwi = 188064.81655062342
Re = 1.4999822619982095
Ri = 0.55089086571081913
Rw = 5.6456475609117183
Rg = 3.9933826145529263
A, B, C, D = get_ABCD(Cz, Cwe, Cwi, Re, Ri, Rw, Rg)

d = disturbances_dt.values[:,:5]

# solve ode
f = lambda t, y, args: zone_state_space(t, y, *args)#args[0], args[1], args[2]) 

x0 = jnp.array([20., 30., 26.])
dx0 = f(0, y=x0, args=(A, B, d[0,:]))
print(dx0.shape)
print(dx0)


ts = 0
te = 86400*30
dt = 900
term = ODETerm(f)
solver = Dopri5()
saveat = SaveAt(ts = range(ts, te+1, dt))
#stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
solution = diffeqsolve(term, solver, t0=ts, t1=te, dt0=dt, y0=x0, saveat=saveat, args=(A, B, d[0,:]))

# get some plot
ts = solution.ts
xs = solution.ys

plt.figure()
plt.plot(ts, xs[:, 0], label = 'Tz')
plt.plot(ts, xs[:, 1], label = 'Twe')
plt.plot(ts, xs[:, 2], label = 'Twi')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
#plt.show()


## Using for loop to update the disturbance every time step
def forward(func, ts, te, dt, x0, solver, args):
    # unpack args
    A, B, d = args

    # ode formulation
    term = ODETerm(func)

    # initial step
    tprev = ts
    tnext = ts + dt 
    dprev = d[0,:]
    args = (A, B, dprev)
    state = solver.init(term, tprev, tnext, x0, args)

    # initialize output
    t_all = [tprev]
    x_all = jnp.array([x0])

    # main loop
    i = 0
    x = x0
    while tprev < te:
        x, _, _, state, _ = solver.step(term, tprev, tnext, x, args, state, made_jump=False)
        tprev = tnext 
        tnext = min(tprev + dt, te)

        # update disturbance for next step
        i += 1
        dnext = d[i,:]
        args = (A,B,dnext)

        # append results
        t_all.append(tnext)
        x_att = x.reshape(1,-1)
        x_all = jnp.concatenate([x_all, x_att], axis=0)
    
    return t_all, x_all 

ts = 0
te = 86400*24
dt = 3600
x0 = jnp.array([20., 30., 26.])
#solver = Dopri5()
solver = Euler()
args = (A, B , d)

# sovle with timer
st = time.process_time()
t_all, x_all = forward(f, ts, te, dt, x0, solver, args)
et = time.process_time()
print(f"execution time is: {et - st}s") 

# plot
plt.figure()
plt.plot(t_all, x_all[:, 0], label = 'Tz')
plt.plot(t_all, x_all[:, 1], label = 'Twe')
plt.plot(t_all, x_all[:, 2], label = 'Twi')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
# plt.show()
