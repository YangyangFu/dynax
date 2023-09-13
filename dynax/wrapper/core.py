from __future__ import annotations

from typing import Tuple, Union, Optional, TypeVar, SupportsFloat, Any, Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax
from flax import struct
import flax.linen as nn

from dynax.simulators.simulator import DifferentiableSimulator

Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey
Parameter = flax.core.FrozenDict[str, Array]

ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")
RenderFrame = TypeVar("RenderFrame")

@struct.dataclass
class EnvStates:
    time: SupportsFloat

class Env(nn.Module):
    """ Generic differentiable environment.
    """
    start_time:float 
    end_time:float
    dt: float 

    def setup(self):
        """Sets up the environment by specifying a differentiable model
        """
        self.init_states: EnvStates
        self.simulator: DifferentiableSimulator
        raise NotImplementedError

    def __call__(self, states: EnvStates, action: ActionType) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any], EnvStates]:

        raise NotImplementedError
    
    def step(
        self,
        states: EnvStates,
        action: ActionType,
        params: Parameter,
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any], EnvStates]:
        """ Run one step of the environment dynamics.

        Args:
            key: random key, used for stochastic environments. Not supported yet.
            state: current state
            action: action to take
            params: parameters of the environment

        Returns:
            obs_next: next observation
            reward: reward
            terminated: whether the episode is terminated
            truncated: whether the episode is truncated
            info: additional information
            states_next: next state. JAX is stateless design, so we need to return the next state explicitly.
        """

        # Use default env parameters if no others specified
        key, key_reset = jax.random.split(key)
        obs_next, reward, terminated, truncated, info, states_next = self.apply(
            params, states, action
        )
        
        # TODO: auto-reset environment based on termination
        # obs_re, states_re = self.reset_env(key_reset, params)

        return obs_next, reward, terminated, truncated, info, states_next

    def reset(
            self, 
            key: PRNGKey,
            params: Parameter,
            states_init: Optional[EnvStates] = None 
        ) -> Tuple[ObsType, EnvStates]:
        """Performs resetting of environment.
        """

        obs, state = self._reset_(key, params, states_init)
        return obs, state

    def _reset_(
            self, 
            key: PRNGKey, 
            params: Parameter,
            states_init: Optional[EnvStates] = None
        ) -> Tuple[ObsType, EnvStates]:
        """Specify the resetting of customized environment.
        """
        raise NotImplementedError

    def _get_obs(self, states: EnvStates) -> ObsType:
        """Applies observation function to state.
        """
        raise NotImplementedError

    def is_terminal(self, states: EnvStates, params: Parameter) -> bool:
        """Checks whether state transition is terminal.
        """
        raise NotImplementedError

    def discount(self, states: EnvStates, params: Parameter) -> float:
        """Return a discount of zero if the episode has terminated.
        """
        return jax.lax.select(self.is_terminal(states, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name.
        """
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self, params: Optional[Parameter]=None):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params: Optional[Parameter]=None):
        """Observation space of the environment."""
        raise NotImplementedError

    def render(self, frame: RenderFrame) -> None:
        """Renders environment."""
        raise NotImplementedError
    
#class Wrapper(Union[Env, DiffEnv]):
#    ...