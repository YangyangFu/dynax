import jax.numpy as jnp
import flax.linen as nn

from .base import BaseAgent

class Tabular(BaseAgent):
    """ Tabular agent 
        The control actions are stored in a table by time step and state if any.
        
        This agent can be used with predefined control schedules.
        
        Args:
            states: jnp.ndarray, a table of states by time step and state if any
            actions: jnp.ndarray, a table of actions by time step and state if any.
            mode: str, the mode of the agent
                - 'constant': the agent will use piecewise constant control actions
                - 'linear': the agent will use a linear interpolation between the states
                - other interpolation modes can be added later
            
        Usage:
            agent = Tabular()
    """
    states: jnp.ndarray
    actions: jnp.ndarray
    interpolation: nn.Module
    
    def __call__(self, state):
        """ return the action given the state """
        return self.interpolation.evaluate(self.states, self.actions, state)
        

