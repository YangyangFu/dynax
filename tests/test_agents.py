import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.tree_util import Partial
import numpy as np 

from dynax.utils.interpolate import PiecewiseConstantInterpolation, LinearInterpolation
from dynax.agents import Tabular

def test_tabular():
    ts = jnp.array([0.0, 2.0, 3.0, 4.0])
    xs = jnp.array([[0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 6.0, 12.0]]).T
    t = jnp.array([0.5, 1.5, 2.5, 3.5]).reshape(-1,1)
    
    # constant interpolation
    const = Tabular(ts=ts, xs=xs, mode='constant')
    init_params = const.init(jax.random.PRNGKey(0), t)
    xt = const.apply(init_params, t)
    assert np.allclose(xt, jnp.array([[ 0., 0.], [ 1., 2.], [ 4., 6.], [ 9., 12.]])), "constant interpolation failed in Tabular agent"
    
    # linear interpolation
    linear = Tabular(ts=ts, xs=xs, mode='linear')
    init_params = linear.init(jax.random.PRNGKey(0), t)
    xt = linear.apply(init_params, t)
    assert np.allclose(xt, jnp.array([[0.25, 0.5], [0.75, 1.5], [2.5, 4.], [6.5, 9.]])), "linear interpolation failed in Tabular agent"
    
if __name__ == "__main__":
    print("Running tests for dynax.agents...")
    test_tabular()
    
    print("All tests passed!")