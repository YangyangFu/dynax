import abc
from typing import Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.base_block_state_space import BaseBlockSSM
from ..agents.tabular import Tabular
from ..utils.interpolate import PiecewiseConstantInterpolation, LinearInterpolation

Scalar = float

# TODO:
# 1. how to best handle the state update? the state update is always 1 step ahead of the output. This will give a problem when used for MPC as a forward simulator.
# 2. is this design modular enough to handle different scenarios of input models? e.g. neural network, tabular, etc.
# 3. how to simulate neural policy?. For model-based control, the simulator will serve as model inside the policy, which will take control inputs from the policy. For model-free control, the simulator should serve as a virtual env. Therefore, there is no need to specify a neural policy here.
# 4. but u contains disturbances, which could be a neural net model. How to cosimulate?

class DifferentiableSimulator(nn.Module):
    model: nn.Module
    disturbance: Union[Scalar, nn.Module]
    agent: Union[Scalar, jax.Array, Tabular]
    estimator: Optional[Union[Scalar, nn.Module]]
    start_time: Scalar = 0
    end_time: Scalar = 1.
    dt: Scalar = 1.
    
    @abc.abstractmethod
    def _gather_model_inputs(self, u: Union[Scalar, jax.Array, nn.Module], d: Union[Scalar, jax.Array, nn.Module]) -> jax.Array:
        """ Gather inputs for dynamic model

            used for rearranging the dynamic model inputs if separate models for control inputs and disturbance inputs are used.

            ```
            ut = Policy(ut_inputs)
            dt = Disturbance(dt_inputs)
            inputs = stack(ut,dt) # any customized order is supported
            x_next, y = model(x, inputs)

        """
        # by default, we use (u ,d)
        return jnp.hstack((u,d))
    

    def setup(self):
        tsol = jnp.arange(self.start_time, self.end_time + self.dt, self.dt)


    @nn.compact
    def __call__(self, x_init: Union[Scalar, jax.Array]):
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
        if isinstance(x_init, Scalar):
            assert self.model.state_dim == 1

        # specify output intervals: in case different time of points are of interest
        tsol = jnp.arange(self.start_time, self.end_time + self.dt, self.dt)

        # scan function
        def roll(carry, ts):
            t, xt = carry

            # action and disturbance
            # 
            ut = agent(t)
            dist = disturbance(t)
            inputs = self._gather_model_inputs(ut, dist)
            inputs = inputs.reshape((self.model.input_dim,))
            # forward simulation
            # \dot x(t) = f(x(t), u(t))
            # y(t) = g(x(t), u(t))
            
            xt_rhs, yt = self.model(xt, inputs)
            
            # TODO: specify a solver object
            # explicit Euler
            # x(t+1) = x(t) + dt * \dot x(t)
            x_tnext = xt + xt_rhs*self.dt
            tnext = t + self.dt
            return (tnext, x_tnext), (xt, yt)
        
        # standardize type
        agent = self.agent
        if isinstance(self.agent, Scalar):
            agent = Tabular(tsol, self.agent*jnp.ones_like(tsol).reshape(-1,1), interpolation=LinearInterpolation())
        elif isinstance(self.agent, jax.Array):
            # have to be rank 1 
            assert len(self.agent.shape) == 1
            agent = self.agent.reshape(1, -1)
            agent = jnp.tile(agent, (len(tsol), 1))
            agent = Tabular(tsol, agent, interpolation=LinearInterpolation())

        disturbance = self.disturbance
        if isinstance(self.disturbance, Scalar):
            disturbance = Tabular(tsol, self.disturbance*jnp.ones_like(tsol).reshape(-1,1),interpolation=LinearInterpolation())  
            self._disturbance = disturbance
        elif isinstance(self.disturbance, jax.Array):
            # have to be rank 1 
            assert len(self.disturbance.shape) == 1
            disturbance = self.disturbance.reshape(1, -1)
            disturbance = jnp.tile(disturbance, (len(tsol), 1))
            disturbance = Tabular(tsol, disturbance, interpolation=LinearInterpolation())

        # Flax modules has to be called before using jax.lax.scan()
        # TODO: try linen.scan() 
        _, _ = self.model(x_init, jnp.zeros((self.model.input_dim, )))
        
        #a_inits = agent.init(jax.random.PRNGKey(0), self.start_time)
        #d_inits = disturbance.init(jax.random.PRNGKey(0), self.start_time)
        agent(self.start_time)
        disturbance(self.start_time)
        # main simulation loop
        carry_init = (self.start_time, x_init)
        # self.t[:-1] is used to avoid the last time step
        (_, x_final), (xsol, ysol) = jax.lax.scan(roll, carry_init, tsol)

        return tsol, xsol, ysol

