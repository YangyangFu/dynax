import jax
import jax.numpy as jnp

from simulator import DifferentiableSimulator
from dynax.models.RC import Continuous4R3C
from dynax.agents import Tabular

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
x0 = jnp.array([20., 30., 26.]) # initial state

RESULTS = jnp.array(
    [[ 2.0000e+01,3.0000e+01,2.6000e+01],
    [ 9.0000e+00,-2.0000e+00,2.5000e+01],
    [ 1.9000e+01,2.9000e+01,-1.7000e+01],
    [-3.3000e+01,-4.4000e+01,6.6000e+01],
    [ 1.0200e+02,1.1200e+02,-1.4200e+02],
    [-2.4100e+02,-2.5200e+02,3.5700e+02],
    [ 6.0100e+02,6.1100e+02,-8.4900e+02],
    [-1.4470e+03,-1.4580e+03,2.0620e+03],
    [ 3.5120e+03,3.5220e+03,-4.9660e+03],
    [-8.4750e+03,-8.4860e+03,1.2001e+04],
    [ 2.0479e+04,2.0489e+04,-2.8961e+04]]
)

# inputs
# disturbance sequence
t = jnp.arange(n_steps+1)*dt + ts
dist = jnp.ones((n_steps+1,4))
policy = jnp.ones((n_steps+1,1))

# gather for RC model inputs
inputs = jnp.concatenate([dist[:,:2], policy, dist[:, 2:]], axis=1)
inputs = Tabular(ts=t, xs=inputs, mode='linear')


# simulator
simulator = DifferentiableSimulator(
    model = model,
    dt = dt
)

# need initialize the simulator first
inits = simulator.init(jax.random.PRNGKey(0), inputs, x0, ts, te)

# simulate forward problem
_, xsol, ysol = simulator.apply(inits, inputs, x0, ts, te)
#_, xsol, ysol = simulator(x0)
assert jnp.allclose(xsol, RESULTS), "closed loop test didn't get expected results."


def test_closed_loop():

    t = ts
    res = [x0] 
    x_prev = x0

    simulator = DifferentiableSimulator(
    model = model,
    dt = dt,
    )
    inits = simulator.init(jax.random.PRNGKey(0), jnp.array([1., 1., 1., 1., 1.]), x_prev, ts, te)

    while t < te:
        dist = jnp.array([1., 1., 1., 1.])
        policy = 1.
        inputs = jnp.concatenate([dist, jnp.array(policy).reshape(-1)])

        _, xsol, ysol = simulator.apply(inits, inputs, x_prev, t, t+dt)

        x_prev = xsol[-1,:]
        res.append(x_prev)
        t += dt 
    
    res = jnp.stack(res, axis=0)

    assert jnp.allclose(res, RESULTS), "closed loop test didn't get expected results."
    return res

import time
start = time.time()
test_closed_loop().block_until_ready()
end = time.time()
print(end-start)