import jax
import jax.numpy as jnp
import flax 
from flax import struct
from functools import partial
import flax.linen as nn

class EnvStates(flax.struct.PyTreeNode):
    x: jnp.ndarray # [Tout, Text_wall, Tint_wall] 
    time: float 

    def update(self, x: jnp.ndarray, time: float = 0.0, **kwargs):
        return self.replace(
            x= x,
            time=time,
            **kwargs,
        )

    @classmethod
    def create(cls, x: jnp.ndarray, time: float = 0.0, **kwargs):
        return cls(x=x, time=time, **kwargs)
    
state = EnvStates.create(x=jnp.array([20., 30., 26.]), time=0.)
print(state)
new_state = state.update(x=jnp.array([21,21,21]), time=1.)
print(state)
print(new_state)

# test jittable conditions
def out_of_bound(x):

    return x - 2.0

x = jnp.array([20., 30., 26.])
#print(jax.jit(out_of_bound, static_argnames=('x'))(x))


#@partial(jax.jit, static_argnums=(0,))
def out_of_bound(x):
    y = nn.activation.relu(x - 40.).sum()

    res = jax.lax.cond(y > 0.0, 
                 lambda x: 1.0,
                 lambda x: 0.0,
                 y)
    
    return res


x = jnp.array([20., 30., 26.])
print(dir(x))
print(out_of_bound(x))
jitted = jax.jit(out_of_bound)
print(jitted(x))