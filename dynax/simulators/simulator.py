import flax.linen as nn
import jax
import jax.numpy as jnp
from ..core.base_block_state_space import BaseBlockSSM

# TODO:
# 1. how to best handle the state update? the state update is always 1 step ahead of the output. This will give a problem when used for MPC as a forward simulator.
# 2. is this design modular enough to handle different scenarios of input models? e.g. neural network, tabular, etc.
# 3. how to simulate neural policy?. For model-based control, the simulator will serve as model inside the policy, which will take control inputs from the policy. For model-free control, the simulator should serve as a virtual env. Therefore, there is no need to specify a neural policy here.
# 4. but u contains disturbances, which could be a neural net model. How to cosimulate?

class DifferentiableSimulator(nn.Module):
    model: nn.Module
    dt: float
    time: float = 0 

    @nn.compact
    def __call__(self, x_init, u, start_time, end_time):
        """ Differentiable simulator for a given model and simulation settings. 

        Args:
            x_init (jnp.ndarray): initial state
            u (tabular policy): ut = u(t)
            tsol (jnp.ndarray): time vector for outputs
        
        Returns:
            xsol (jnp.ndarray): state trajectory
            ysol (jnp.ndarray): output trajectory

        """

        def scan_fn(carry, ts):
            i, xi = carry

            # or ui = u.act(t)
            # TODO: this has to be a control agent model
            # how to deal with a mixture of agents (neural policy for control, tabular for disturbance)
            ui = u[i,:]

            # forward simulation
            # \dot x(t) = f(x(t), u(t))
            # y(t) = g(x(t), u(t))
            xi_rhs, yi = self.model(xi, ui)
            
            # TODO: specify a solver object
            # explicit Euler
            # x(t+1) = x(t) + dt * \dot x(t)
            x_next = xi + xi_rhs*self.dt

            return (i+1, x_next), (xi, yi)
        
        # specify solver intervals

        # specify output intervals: in case different time of points are of interest
        # TODO: need make sure the last time instance is the end time
        tsol = jnp.arange(start_time, end_time + self.dt, self.dt)

        # control inputs
        # if u is a scalar, then it is a constant input
        # if u is a tabular policy, then it will take the value of time as input and gernerate a control signal
        
        # module has to be called once before the while loop
        _, _ = self.model(x_init, jnp.zeros_like(u[0]))

        # initialize model for observations at start time
        #y_init = self.model._call_observation(x_init, jnp.zeros_like(u[0,:]))

        # main simulation loop
        u = u.reshape(-1, self.model.input_dim)   
        carry_init = (0, x_init)
        # self.t[:-1] is used to avoid the last time step
        (_, x_final), (xsol, ysol) = jax.lax.scan(scan_fn, carry_init, tsol)

        # append initial point for y and final point for x
        #xsol = jnp.concatenate((xsol, x_final.reshape(1,-1)), axis = 0)
        #ysol = jnp.concatenate((y_init.reshape(1,-1), ysol), axis = 0)
        # TODO: interpolate outputs from solver to match the time vector
        #assert xsol.shape[0] == len(tsol)
        #assert ysol.shape[0] == len(tsol)

        # module attributes are frozen outside of setup()
        #self.time = end_time 

        return tsol, xsol, ysol

