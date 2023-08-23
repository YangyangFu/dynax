import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from ..core.base_block_state_space import BaseBlockSSM

# TODO:
# 1. how to best handle the state update? the state update is always 1 step ahead of the output. This will give a problem when used for MPC as a forward simulator.
# 2. is this design modular enough to handle different scenarios of input models? e.g. neural network, tabular, etc.
# 3. how to simulate neural policy?. For model-based control, the simulator will serve as model inside the policy, which will take control inputs from the policy. For model-free control, the simulator should serve as a virtual env. Therefore, there is no need to specify a neural policy here.
# 4. but u contains disturbances, which could be a neural net model. How to cosimulate?
from flax import core 
from flax import struct
class SimulatorState(struct.PyTreeNode):
    clock: float
    model_state: jnp.array = struct.field(pytree_node=True)
    
    def update(self, clock, model_state, **kwargs):
        return self.replace(clock = clock, 
                            model_state = model_state,
                            **kwargs)
    @classmethod
    def create(cls, *, clock, model_state, **kwargs):
        return cls(clock=clock,
                   model_state=model_state,
                   **kwargs)

class DifferentiableSimulator(nn.Module):
    state: nn.Module
    model: nn.Module
    disturbance: nn.Module
    agent: nn.Module
    estimator: nn.Module = None
    start_time: float = 0.
    end_time: float = 1.
    dt: float = 1.
    
    @nn.compact
    def __call__(self):
        """ Differentiable simulator for a given model and simulation settings. 

        Args:
            x_init (jnp.ndarray): initial state
            u (tabular policy): ut = u(t)
            tsol (jnp.ndarray): time vector for outputs
        
        Returns:
            xsol (jnp.ndarray): state trajectory
            ysol (jnp.ndarray): output trajectory

        """
        # initialize some calls
        #_, _, = self.model(jnp.zeros((self.model.state_dim,1)), jnp.zeros((self.model.input_dim, #1)))

        # specify solver intervals

        # specify output intervals: in case different time of points are of interest
        # TODO: need make sure the last time instance is the end time
        tsol = jnp.arange(self.start_time, self.end_time + self.dt, self.dt)

        # control inputs
        # if u is a scalar, then it is a constant input
        # if u is a tabular policy, then it will take the value of time as input and gernerate a control signal
        xsol = jnp.zeros((len(tsol), self.model.state_dim))
        ysol = jnp.zeros((len(tsol), self.model.output_dim))
        
        for i, t in enumerate(tsol):
            dist_t = self.disturbance(t)
            u_t = self.agent(t)
            # combine u and disturbance to make model inputs
            # u: (1, nu)
            # dist: (1, du)
            inputs = jnp.hstack((dist_t[:,:2], u_t, dist_t[:, 2:])).reshape(-1,1)
            x_next, y_t = self.model(self.state.model_state, inputs)
            self.state.update(clock = t, 
                              model_state = x_next
                              )
            # save results
            xsol = xsol.at[i,:].set(x_next.reshape(-1,))
            ysol = ysol.at[i,:].set(y_t.reshape(-1,))
        # module has to be called once before the while loop
        #_, _ = self.model(x_init, jnp.zeros_like(u(self.start_time))

        # initialize model for observations at start time
        #y_init = self.model._call_observation(x_init, jnp.zeros_like(u[0,:]))
        
        return tsol, xsol, ysol

