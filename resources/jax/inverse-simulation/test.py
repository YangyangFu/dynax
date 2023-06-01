import jax 
import jax.numpy as jnp 
from typing import Sequence
import flax.linen as nn

class MLP(nn.Module):
  features: int
  tol : float
  max_iter : int
  affine_map: nn.Module

#  def setup(self):
#    self.affine_map = nn.Dense(self.features)

  #@nn.compact
  def __call__(self, x):

    # Repeated Layer
    #affine_map = nn.Dense(self.features)
    layer = lambda z: nn.tanh(self.affine_map(z) + x)

    # Fwd Pass
    def cond_fun(carry):
      z_prev, z, _ = carry 
      return jnp.linalg.norm(z_prev-z) > self.tol 
    
    def body_fun(carry):
      _, z, iter = carry 
      return z, layer(z), iter + 1
    
    init_carry = (jnp.zeros_like(x), layer(jnp.zeros_like(x)), 0)
    return  jax.lax.while_loop(cond_fun, body_fun, init_carry)[1:]

data_dim = 10 
affine = nn.Dense(data_dim)
model = MLP(data_dim, 1e-4, 1000, affine)
x = jax.random.normal(jax.random.PRNGKey(1), shape=(1, data_dim))
variables = model.init(jax.random.PRNGKey(0), x)
output = model.apply(variables, x)
print(output)
print(model.tabulate(jax.random.PRNGKey(0), x))
print(variables)