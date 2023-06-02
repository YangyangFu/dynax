import jax 
import jax.numpy as jnp 
import flax.linen as nn
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 

from dynax.models.RC import Discrete4R3C
from dynax.models.RC import Continuous4R3C

# a simple RC zone model -> ODE system
"""
===================================================
This is a 4r3c model for zone temperature prediction
ODE::   xdot = Ax + Bu
        y = Cx
States:
        x = [Tai, Twe, Twi]'
Disturbances:
        u = [Tao, qCon_i, qHVAC, qRad_e, qRad_i]'
Output:
        y = [Tai]
===================================================
"""

n_devices = jax.local_device_count()

#model = Discrete4R3C()
model = Continuous4R3C()
state_dim = model.state_dim
input_dim = model.input_dim
output_dim = model.output_dim

# investigate the model structure
print(model.tabulate(jax.random.PRNGKey(0), jnp.zeros((state_dim,)), jnp.zeros((input_dim,))))

# load model parameters: Cai, Cwe, Cwi, Re, Ri, Rw, Rg
params = [10384.31640625, 499089.09375, 1321535.125,
        1.5348844528198242, 0.5000327825546265, 1.000040054321289, 
        20.119935989379883]

rc = {'Cai': params[0], 'Cwe': params[1], 'Cwi': params[2],
    'Re': params[3], 'Ri': params[4], 'Rw': params[5], 'Rg': params[6]
    }
params = {'params': rc}

# define a forward step function
@jax.jit
def forward_step(params, state, input):
    dx, output = model.apply(params, state, input)
    new_state = state + dx*dt
    return new_state, output

# load input data
inputs = pd.read_csv('./data/eplus_1min.csv', index_col=[0])
n_samples = len(inputs)
index = range(0, n_samples*60, 60)
inputs.index = index

# resample to a given time step
dt = 900
inputs_dt = inputs.groupby([inputs.index // dt]).mean()
u_dt = inputs_dt.values[:,:5]
y_dt = inputs_dt.values[:,5] 

# simulate the model for 100 steps
# initial state
state = jnp.array([20., 30., 26.])  

# initialize output
outputs = []

# main loop
n_steps = len(inputs_dt)
i = 0

ts = time.time()
while i < n_steps:
    # random input
    ut = u_dt[i,:]

    # advance one step
    new_state, output = forward_step(params, state, ut)

    # update state
    state = new_state
    i += 1

    # save output
    outputs.append(output)
te = time.time()
print('time elapsed with jit: ', te-ts)

# plot the results  
plt.figure()
plt.plot(y_dt[:n_steps], label='y')
plt.plot(outputs, label='y_hat')
plt.legend()
plt.savefig('forward_simulation.png')
plt.close()

