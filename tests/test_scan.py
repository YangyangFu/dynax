import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.tree_util import Partial
import numpy as np 
from typing import Any
from jax.scipy.ndimage import map_coordinates

from dynax.agents.base import BaseAgent
from dynax.utils.interpolate import PiecewiseConstantInterpolation, LinearInterpolation

class LRNNCell(nn.Module):
    @nn.compact
    def __call__(self, h, x, x1):
        nh = h.shape[-1]
        Whx = nn.Dense(nh)
        Whh = nn.Dense(nh, use_bias=False)
        Wyh = nn.Dense(1)
        x = x1 + x 

        h = nn.tanh(Whx(x) + Whh(h))
        y = nn.tanh(Wyh(h))
        return h, y

class LRNN(nn.Module):
    ny: Any
    nh: Any

    @nn.compact
    def __call__(self, x, x1):
        # 
        cell = nn.scan(
            LRNNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes = 1,
            out_axes = 1,
        )
        b = x.shape[0]
        h = jnp.zeros((b, self.nh))
        h, y = cell()(h, x[..., None], x1[..., None])
        return y

x = jnp.array([[1,2,3,4,5,6], [5,6,7,8,9,10]])
#x = x[0, :]
print(x.shape)
x1 = x
lrnn = LRNN(ny=1, nh=2)
init_params = lrnn.init(jax.random.PRNGKey(0), x, x1)
print(init_params)
y = lrnn.apply(init_params, x, x1)
print(y)
print(y.shape)

class Tabular(BaseAgent):
    @nn.compact
    def __call__(self, states, actions, state, mode='linear'):
        if mode == 'linear':
            interp = LinearInterpolation(ts=states, xs=actions)
        else:
            interp = LinearInterpolation(ts=states, xs=actions)

        y = interp(at = state)
        
        return y


ts = jnp.array([0.0, 2.0, 3.0, 4.0])
# (4,2)
xs = jnp.array([[0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 6.0, 12.0]]).T
t = jnp.array([0.5, 1.5])

class Wrapper(nn.Module):
    @nn.compact 
    def __call__(self, ts, xs, t, mode='linear'):
        
        # if the module does not have a carry, a scan wrapper is needed.
        def scan_fcn(tab, carry, tmp):
            y = tab(ts, xs, t)
            return carry+1, y
        
        tab = Tabular()
        carry = jnp.zeros(1)
        # do 10 times
        tmp = jnp.arange(10)
        scan = nn.scan(scan_fcn, 
                       variable_broadcast='params',
                       split_rngs={'params':False},
                       in_axes=0,
                       out_axes=0,)

        carry, y = scan(tab, carry, tmp)

        return y

tab = Tabular()
init_params = tab.init(jax.random.PRNGKey(0), ts, xs, t)
y = tab.apply(init_params, ts, xs, t)

# test wrapper
wrap = Wrapper()
init_params = wrap.init(jax.random.PRNGKey(0), ts, xs, t)
y = wrap.apply(init_params, ts, xs, t)
print(y.shape)

def fcn(t):
    y = wrap.apply(init_params, ts, xs, t)
    return jnp.sum(y)

y, grads = jax.value_and_grad(fcn)(t)
print(y, y.shape)
print(grads, grads.shape)