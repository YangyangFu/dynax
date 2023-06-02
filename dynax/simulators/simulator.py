import flax.linen as nn
import jax
import jax.numpy as jnp
from ..systems.base_block_state_space import BaseBlockSSM

# TODO:
# 1. how to best handle the state update? the state update is always 1 step ahead of the output. This will give a problem when used for MPC as a forward simulator.
class DifferentiableSimulator(nn.Module):
    model: nn.Module
    t: jnp.ndarray
    dt: float

    @nn.compact
    def __call__(self, x_init, u):
     
        def scan_fn(carry, ts):
            i, xi = carry
            ui = u[i,:]

            # forward simulation
            # \dot x(t) = f(x(t), u(t))
            # y(t) = g(x(t), u(t))
            xi_rhs, yi = self.model(xi, ui)
            
            # explicit Euler
            # x(t+1) = x(t) + dt * \dot x(t)
            x_next = xi + xi_rhs*self.dt

            return (i+1, x_next), (xi, yi)

        # module has to be called once before the while loop
        _, _ = self.model(x_init, jnp.zeros_like(u[0,:]))

        # main simulation loop
        u = u.reshape(-1, self.model.input_dim)   
        carry_init = (0, x_init)
        # self.t[:-1] is used to avoid the last time step
        (_, x_final), (xsol, ysol) = jax.lax.scan(scan_fn, carry_init, self.t)

        assert xsol.shape[0] == len(self.t)
        assert ysol.shape[0] == len(self.t)

        return xsol, ysol

