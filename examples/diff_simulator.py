import jax 
import jax.numpy as jnp
import pandas as pd 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 

from dynax.models.RC import Continuous4R3C
from dynax.simulators.simulator import DifferentiableSimulator

# instantiate a model
#model = Discrete4R3C()
model = Continuous4R3C()
state_dim = model.state_dim
input_dim = model.input_dim
output_dim = model.output_dim

# load data
inputs = pd.read_csv('./data/eplus_1min.csv', index_col=[0])
n_samples = len(inputs)
index = range(0, n_samples*60, 60)
inputs.index = index

# resample to a given time step
dt = 900
inputs_dt = inputs.groupby([inputs.index // dt]).mean()
u_dt = inputs_dt.values[:,:5]
y_dt = inputs_dt.values[:,5] 

# forward simulation settings
tsol = jnp.arange(0, len(u_dt)*dt, dt)
state = jnp.array([20., 30., 26.])  # initial state

simulator = DifferentiableSimulator(model, tsol, dt)
print(simulator.tabulate(jax.random.PRNGKey(0), jnp.zeros((model.state_dim,)), u_dt))

# forward simulation
# simulate with given params: Cai, Cwe, Cwi, Re, Ri, Rw, Rg
params = [10384.31640625, 499089.09375, 1321535.125,
        1.5348844528198242, 0.5000327825546265, 1.000040054321289, 
        20.119935989379883]
rc = {'Cai': params[0], 'Cwe': params[1], 'Cwi': params[2],
    'Re': params[3], 'Ri': params[4], 'Rw': params[5], 'Rg': params[6]
    }
params_true = {'params': {'model': rc}}

ts = time.time()
states, outputs = simulator.apply(params_true, state, u_dt)
te = time.time()
print('time elapsed for forward simulation: ', te-ts)

# plot the results  
plt.figure()
plt.plot(y_dt, label='y')
plt.plot(outputs, label='y_hat')
plt.legend()
plt.savefig('diff_simulator.png')
plt.close()
