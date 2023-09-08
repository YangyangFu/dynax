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

    def __checker__(self, inputs, states_init):
        """ Check setup """
        pass

    @nn.compact
    def __call__(self, inputs, states_init: Union[Scalar, jax.Array], start_time: Scalar, end_time: Scalar):

        """ Differentiable simulator for a given model and simulation settings. 

        Args:
            x_init (jnp.ndarray): initial state, (state_dim, ) or (state_dim, 1)
            u (tabular policy): ut = u(t), (input_dim, ) or (input_dim, 1)
            tsol (jnp.ndarray): time vector for outputs
        
        Returns:
            xsol (jnp.ndarray): state trajectory
            ysol (jnp.ndarray): output trajectory

        """
        # TODO: add assertions
        # - check input dimension
        # - check rank: rank of x == rank of u
        if isinstance(states_init, Scalar):
            assert self.model.state_dim == 1

        # specify output intervals: in case different time of points are of interest
        tsol = jnp.arange(start_time, end_time + self.dt, self.dt)
        
        # scan function
        def rollout(model, carry, tsol):
            t, xt = carry

            # action and disturbance
            #
            inputs_t = inputs.apply(init_params_inputs, t) 
            #inputs_t = _inputs(t)
            inputs_t = inputs_t.reshape((model.input_dim,))
            # forward simulation
            # \dot x(t) = f(x(t), u(t))
            # y(t) = g(x(t), u(t))

            xt_rhs, yt = model(xt, inputs_t)
            
            # TODO: specify a solver object
            # explicit Euler
            # x(t+1) = x(t) + dt * \dot x(t)
            xt_next = xt + xt_rhs*self.dt
            tnext = t + self.dt
            return (tnext, xt_next), (xt, yt)
        
        # standardize type
        if isinstance(inputs, Scalar):
            inputs = Tabular(tsol, inputs*jnp.ones_like(tsol).reshape(-1,1), mode=self.mode_interp)
        elif isinstance(inputs, jax.Array):
            # have to be rank 1 
            assert len(inputs.shape) == 1
            inputs = jnp.tile(inputs.reshape(1,-1), (len(tsol), 1))
            inputs = Tabular(tsol, inputs,  mode=self.mode_interp)

        # have to initialize nn.Module from __call__() method
        # not needed if nn.Module is defined within setup() or init
        init_params_inputs = inputs.init(jax.random.PRNGKey(0), start_time)

        # main simulation loop
        carry_init = (start_time, states_init)
        
        scan_roll = nn.scan(rollout,
                            variable_broadcast='params',
                            split_rngs={'params':False},
                            in_axes=0,
                            out_axes=0,
                            )
        (_, x_final), (xsol, ysol) = scan_roll(self.model, carry_init, tsol)

        return tsol, xsol, ysol

