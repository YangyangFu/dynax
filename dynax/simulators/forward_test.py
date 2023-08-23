import jax 
import jax.numpy as jnp 
import pandas as pd 

from dynax.simulators.forward import SimulatorState, DifferentiableSimulator
from dynax.models.RC import Continuous4R3C
from dynax.agents import Tabular
from dynax.utils.interpolate import PiecewiseConstantInterpolation, LinearInterpolation

# instantiate a model
#model = Discrete4R3C()
model = Continuous4R3C()
state_dim = model.state_dim
input_dim = model.input_dim
output_dim = model.output_dim

# resample to a given time step
dt = 1.


# forward simulation settings
ts = 0.
te = 10.
n_steps = int((te - ts) / dt)
x0 = jnp.array([20., 30., 26.]).reshape(-1,1)  # initial state

states = SimulatorState.create(
    clock = 0,
    model_state = x0
)

# disturnace model
dist = Tabular(states=jnp.arange(n_steps)*dt+ts, 
               actions=jnp.ones((n_steps,4)), 
               interpolation=LinearInterpolation())
policy = Tabular(states=jnp.arange(n_steps)*dt+ts, 
               actions=jnp.ones((n_steps,1)), 
               interpolation=LinearInterpolation())

simulator = DifferentiableSimulator(
    state = states,
    model = model,
    disturbance=dist,
    agent=policy,
    estimator=None,
    start_time = ts,
    end_time = te,
    dt = dt
)


#print(simulator.tabulate(jax.random.PRNGKey(0), jnp.zeros((model.state_dim,)), u_dt, ts, te))
# need initialize the simulator first
simulator.init(jax.random.PRNGKey(0))
print(simulator.tabulate(jax.random.PRNGKey(0)))

