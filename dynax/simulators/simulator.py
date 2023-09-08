import abc
from typing import Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from dynax.agents.tabular import Tabular

Scalar = float

class DifferentiableSimulator(nn.Module):
    model: nn.Module
    dt: Scalar = 1.
    mode_interp: str = 'linear'
    start_time: Scalar = 0.

    def __checker__(self, inputs, states_init):
        """ Check setup """
        pass

    @nn.compact
    def __call__(self, states_init: Union[Scalar, jax.Array], inputs: jax.Array):

        """ Differentiable simulator for a given model and simulation settings. 

        This is to mimic RNN style implementation, where the simulation forward tiem steps are determined by the input sequence.
        inputs: (T, N)
        """
        # TODO: add assertions
        # - check input dimension
        # - check rank: rank of x == rank of u
        if isinstance(states_init, Scalar):
            assert self.model.state_dim == 1

        def rollout(model, carry, inputs):
            t, states = carry 
            rhs, y = model(states, inputs)
            # residual connection: explicit euler method
            states += self.dt * rhs
            t += self.dt

            return (t, states), (states, y)

        # main simulation loop        
        scan = nn.scan(rollout,
                        variable_broadcast='params',
                        split_rngs={'params':False},
                        in_axes=0,
                        out_axes=0,
                        )
        carry = (self.start_time, states_init)
        carry, (xsol, ysol) = scan(self.model, carry, inputs)

        return xsol, ysol

