import flax.linen as nn
import jax
import jax.numpy as jnp
from ..examples.base_block_state_space import BaseBlockSSM

# TODO:
# 1. how to best handle the state update? the state update is always 1 step ahead of the output. This will give a problem when used for MPC as a forward simulator.
# 2. is this design modular enough to handle different scenarios of input models? e.g. neural network, tabular, etc.
# 3. how to simulate neural policy?. For model-based control, the simulator will serve as model inside the policy, which will take control inputs from the policy. For model-free control, the simulator should serve as a virtual env. Therefore, there is no need to specify a neural policy here.
# 4. but u contains disturbances, which could be a neural net model. How to cosimulate?

class DifferentiableSimulator(nn.Module):
    model: nn.Module
    disturbance: nn.Module
    control: nn.Module
    start_time: float
    end_time: float
    dt: float
    estimator: nn.Module = None
    clock: float = start_time 

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
        # specify solver intervals

        # specify output intervals: in case different time of points are of interest
        # TODO: need make sure the last time instance is the end time
        tsol = jnp.arange(start_time, end_time + self.dt, self.dt)

        # control inputs
        # if u is a scalar, then it is a constant input
        # if u is a tabular policy, then it will take the value of time as input and gernerate a control signal
        
        # module has to be called once before the while loop
        _, _ = self.model(x_init, jnp.zeros_like(u(start_time)))

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

        self.time = end_time 

        return tsol, xsol, ysol

