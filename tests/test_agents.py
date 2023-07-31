import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.tree_util import Partial
import numpy as np 

from dynax.utils.interpolate import PiecewiseConstantInterpolation, LinearInterpolation
from dynax.agents import Tabular

def test_tabular():
    ts = jnp.array([0.0, 1.0, 2.0, 3.0])
    xs = jnp.array([[0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 6.0, 12.0]]).T
    t = 0.5
    
    # constant interpolation
    const = Tabular(states=ts, actions=xs, interpolation=PiecewiseConstantInterpolation())
    xt = const(t)
    assert np.allclose(xt, jnp.array([1.0, 2.0])), "constant interpolation failed in Tabular agent"
    
    # linear interpolation
    linear = Tabular(states=ts, actions=xs, interpolation=LinearInterpolation())
    xt = linear(t)
    assert np.allclose(xt, jnp.array([0.5, 1.0])), "linear interpolation failed in Tabular agent"
    
if __name__ == "__main__":
    print("Running tests for dynax.agents...")
    test_tabular()
    
    print("All tests passed!")