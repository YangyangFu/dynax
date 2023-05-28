import jax 
import jax.numpy as jnp
from jax.tree_util import Partial
import flax.linen as nn
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dynax.models.RC import Discrete4R3C
from dynax.models.RC import Continuous4R3C

# instantiate a model
#model = Discrete4R3C()
model = Continuous4R3C()
state_dim = model.state_dim
input_dim = model.input_dim
output_dim = model.output_dim

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
    states = jnp.zeros((n_steps, state_dim))
    outputs = jnp.zeros((n_steps, output_dim))
    for i in range(n_steps):
        new_state, output = forward_step(model, params, new_state, inputs[i,:], dt)
        states = states.at[i].set(new_state)
        outputs = outputs.at[i].set(output)

    return states, outputs

# forward simulation
tsol = jnp.arange(0, len(u_dt)*dt, dt)
state = jnp.array([20., 30., 26.])  # initial state

# get an initial guess of the parameters
params = model.init(jax.random.PRNGKey(0), state, u_dt[0,:])

# simulate the model
states, outputs = forward(model, params, state, u_dt,  tsol, dt)
