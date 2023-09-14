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
        action: ActionType,
        states: EnvStates,
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
        obs_next, reward, terminated, truncated, info, states_next = self.apply(
            params, action, states
        )
        
        # TODO: auto-reset environment based on termination
        # obs_re, states_re = self.reset_env(key_reset, params)

        return obs_next, reward, terminated, truncated, info, states_next

    def reset(
            self, 
            key: PRNGKey,
            params: Parameter,
            states_init: Optional[EnvStates] = None,
            deterministic: bool = True,
        ) -> Tuple[ObsType, EnvStates, Parameter]:
        """Performs resetting of environment.
        """

        raise NotImplementedError

    @property
    def id(self) -> str:
        """ Environment ID.
        """
        return self.name
    
    @property
    def action_space(self, params: Optional[Parameter]=None):
        """Action space of the environment."""
        raise NotImplementedError

    @property
    def observation_space(self, params: Optional[Parameter]=None):
        """Observation space of the environment."""
        raise NotImplementedError

    def render(self, frame: RenderFrame) -> None:
        """Renders environment."""
        raise NotImplementedError
    
#class Wrapper(Union[Env, DiffEnv]):
#    ...