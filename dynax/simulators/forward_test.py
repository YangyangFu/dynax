import jax 
import jax.numpy as jnp 
import pandas as pd 

from dynax.simulators.forward import DifferentiableSimulator
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

def test_tabular_agent():
    # agent model
    dist = Tabular(states=jnp.arange(n_steps+1)*dt+ts, 
                actions=jnp.ones((n_steps+1,4)), 
                interpolation = LinearInterpolation())
    policy = Tabular(states=jnp.arange(n_steps+1)*dt+ts, 
                actions=jnp.ones((n_steps+1,1)), 
                interpolation = LinearInterpolation())

    simulator = DifferentiableSimulator(
    #    state = states,
        model = model,
        disturbance=dist,
        agent=policy,
        estimator=None,
        start_time = ts,
        end_time = te,
        dt = dt
    )

    # need initialize the simulator first
    inits = simulator.init(jax.random.PRNGKey(0), x0)

    # simulate forward problem
    _, xsol, ysol = simulator.apply(inits, x0)

    assert jnp.allclose(xsol, RESULTS), "tabular agent didn't get expected results."


def test_array_agent():
    dist = jnp.array([1., 1., 1., 1.])
    policy = 1.

    simulator = DifferentiableSimulator(
    #    state = states,
        model = model,
        disturbance=dist,
        agent=policy,
        estimator=None,
        start_time = ts,
        end_time = te,
        dt = dt
    )

    # need initialize the simulator first
    inits = simulator.init(jax.random.PRNGKey(0), x0)

    # simulate forward problem
    _, xsol, ysol = simulator.apply(inits, x0)

    assert jnp.allclose(xsol, RESULTS), "tabular agent didn't get expected results."


if __name__ == "__main__":
    test_tabular_agent()
    test_array_agent()
    print("all test in forward simulator passed !!!!")