import flax.linen as nn
import jax.numpy as jnp
from ..systems.base_block_state_space import BaseBlockSSM

# TODO:
# 1. append to list is faster than jnp.append()/vstack()/hstack()
# 2. the current framework for observation function is one step lagged, which is not consistent with the current framework for state function
class DifferentiableSimulator(nn.Module):
    model: BaseBlockSSM
    t: jnp.ndarray
    dt: float

    def __call__(self, x_init, u):
        # NOTE: append to list is faster than jnp.append()/vstack()/hstack()
        xsol = []
        ysol = [] #jnp.array([]).reshape(0, self.model.output_dim)
        xi = x_init
        xsol.append(xi)
        u = u.reshape(-1, self.model.input_dim)

        for i in range(len(self.t)):
            
            xi_rhs, yi = self.model(xi, u[i,:])

            # explicit Euler
            xi = xi + xi_rhs*self.dt

            # save results by appending to list, which is 100x faster than jnp.append() or indexing
            #xsol = xsol.at[i+1].set(xi)
            #ysol = ysol.at[i].set(yi)
            xsol.append(xi)
            ysol.append(yi)
        
        # return list directly is faster than jnp.array()
        return xsol, ysol

