from typing import Callable
import jax 
import jax.numpy as jnp
from jax.tree_util import Partial
import flax.linen as nn
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flax.training.train_state import TrainState
import optax 
import time 

from dynax.models.RC import Discrete4R3C
from dynax.models.RC import Continuous4R3C

# instantiate a model
#model = Discrete4R3C()
model = Continuous4R3C()
state_dim = model.state_dim
input_dim = model.input_dim
output_dim = model.output_dim

# ===========================================================
# Method 1 for forward simulation: jittable function
# ===========================================================
# investigate the model structure
print(model.tabulate(jax.random.PRNGKey(0), jnp.zeros((state_dim,)), jnp.zeros((input_dim,))))

# load calibration data
inputs = pd.read_csv('./disturbance_1min.csv', index_col=[0])
n_samples = len(inputs)
index = range(0, n_samples*60, 60)
inputs.index = index

# resample to a given time step
dt = 900
inputs_dt = inputs.groupby([inputs.index // dt]).mean()
u_dt = inputs_dt.values[:,:5]
y_dt = inputs_dt.values[:,5] 

# TODO: construct a data loader
# forward step with euler method
@Partial(jax.jit, static_argnums=(0,))
def forward_step(model, params, state, input, dt):
    dx, output = model.apply(params, state, input)
    new_state = state + dx*dt
    return new_state, output

def forward(model, params, state, inputs, t, dt):
    """
    Forward simulation of a given model
    """
    n_steps = len(t)
    new_state = state
    states = [state]
    outputs = []
    for i in range(n_steps):
        new_state, output = forward_step(model, params, new_state, inputs[i,:], dt)
        
        # these two steps are slow: 95% of the time is spent on these two steps
        #states = states.at[i].set(new_state)
        #outputs = outputs.at[i].set(output)
        # list append is much faster but still takes 50% of the time
        states.append(new_state)
        outputs.append(output)

    # jnp.array(list) is very slow
    return jnp.array(states), jnp.array(outputs)

# forward simulation
tsol = jnp.arange(0, len(u_dt)*dt, dt)
state = jnp.array([20., 30., 26.])  # initial state

# get an initial guess of the parameters
key = jax.random.PRNGKey(0)
params = model.init(key, state, u_dt[0,:])

# simulate the model
states, outputs = forward(model, params, state, u_dt,  tsol, dt)
#print(states.shape, outputs.shape)

# simulate with given params: Cai, Cwe, Cwi, Re, Ri, Rw, Rg
params = [10384.31640625, 499089.09375, 1321535.125,
        1.5348844528198242, 0.5000327825546265, 1.000040054321289, 
        20.119935989379883]
rc = {'Cai': params[0], 'Cwe': params[1], 'Cwi': params[2],
    'Re': params[3], 'Ri': params[4], 'Rw': params[5], 'Rg': params[6]
    }
params_true = {'params': rc}

ts = time.time()
states, outputs = forward(model, params_true, state, u_dt,  tsol, dt)
te = time.time()
print('forward simulation with jit takes {} seconds'.format(te-ts))

# =========================================================
# Method 2 for forward simulation: nn.Module
#   - need add jit to forward otherwise slow simulation
#   - 50% slower than Method 1
# =========================================================

# inherite from a nn.Module 
class Simulator(nn.Module):
    model: nn.Module
    t: jnp.ndarray
    dt: float
    
    # TODO: add a solver for ODEs
    def __call__(self, x_init, u):
        xsol = []
        ysol = [] #jnp.array([]).reshape(0, self.model.output_dim)
        xi = x_init
        #xsol = xsol.at[0].set(xi)
        xsol.append(xi)
        u = u.reshape(-1, self.model.input_dim)
        for i in range(len(self.t)):
            ui = u[i,:]
            xi_rhs, yi = self.model(xi, ui)
            # explicit Euler
            xi = xi + xi_rhs*self.dt
            
            # save results
            #xsol = xsol.at[i+1].set(xi)
            #ysol = ysol.at[i].set(yi)
            xsol.append(xi)
            ysol.append(yi)
            
        return jnp.array(xsol), jnp.array(ysol)

simulator = Simulator(model, tsol, dt)
print(simulator.tabulate(jax.random.PRNGKey(0), jnp.zeros((model.state_dim,)), u_dt))

params_true = {'params': {'model': params_true['params']}}
ts = time.time()
states, outputs = simulator.apply(params_true, state, u_dt)
te = time.time()
print('forward simulation without jit takes {} seconds'.format(te-ts))