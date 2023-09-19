import gymnasium as gym

from typing import Any, Protocol, Union, Optional
from dataclasses import dataclass, field

class EnvCreator(Protocol):
    """Function type expected for an environment."""
    def __call__(self, **kwargs: Any) -> Any:
        ...

# TODO: need redesign the vectorization part
class VectorEnvCreator(Protocol):
    """Function type expected for an environment."""

    def __call__(self, **kwargs: Any):
        ...

@dataclass
class EnvParams:
    ...

@dataclass
class WrapperSpec:
    """A specification for recording wrapper configs.

    * name: The name of the wrapper.
    * entry_point: The location of the wrapper to create from.
    * kwargs: Additional keyword arguments passed to the wrapper. If the wrapper doesn't inherit from EzPickle then this is ``None``
    """

    name: str
    entry_point: str
    kwargs: Union[Any, None]

@dataclass
class EnvSpec:
    id: str
    entry_point: Union[EnvCreator, str, None] = field(default=None)

    # environment attri
    reward_threshold: Union[float, None] = field(default=None)
    nondeterministic: bool = field(default=True)

    # wrapper
    max_episode_steps: Union[int, None] = field(default=None)

    # environment arguments
    env_params: Union[EnvParams, None] = field(default=None)

    # post init attributes
    namespace: Union[str, None] = field(init=False) # not generated in __init__()
    name: str = field(init=False)
    version: Union[int, None] = field(init=False)

    # NOT IMPLEMENTED
    # applied wrappers: 

    # vectorized entry point
    

def register(id:str,
             entry_point: Union[EnvCreator, str, None] = None,
             reward_threshold: Union[float, None] = None,
             nondeterministic: bool = False,
             max_episode_steps: Union[int, None] = None,
             env_params: Union[EnvParams, None] = None,
             ):

    pass 


def make(): 
    pass
