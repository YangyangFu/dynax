import jax.numpy as jnp
import flax.linen as nn

from .base import BaseAgent
from ..utils.interpolate import PiecewiseConstantInterpolation, LinearInterpolation

class Tabular(BaseAgent):
    """ Tabular agent 
        The control actions are stored in a table by time step and state if any.
        
        This agent can be used with predefined control schedules.
        
        Args:
            states: jnp.ndarray, a table of states by time step and state if any
            actions: jnp.ndarray, a table of actions by time step and state if any.
            interpolation: str, the mode of the agent
                - 'constant': the agent will use piecewise constant control actions
                - 'linear': the agent will use a linear interpolation between the states
                - other interpolation modes can be added later
            
        Usage:
            agent = Tabular()
    """
    ts: jnp.ndarray
    xs: jnp.ndarray
    mode: str

    #@nn.compact
    #def __call__(self, state):
    #    """ return the action given the state """
    #    return self.interpolation.evaluate(self.states, self.actions, state)
    
    def setup(self):
        if self.mode=="linear":
            interp = LinearInterpolation(ts = self.ts, xs= self.xs)
        elif self.mode == "constant":
            interp = PiecewiseConstantInterpolation(ts = self.ts, xs= self.xs)
        else:
            raise ValueError("Interpolation mode in Tabular has to be either 'linear' or 'constant' !!!")
        self.interp = interp

    def __call__(self,  at):

        return self.interp(at)

