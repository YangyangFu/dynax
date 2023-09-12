from typing import Any, Generic, TypeVar

import chex
import flax.linen as nn 

T = TypeVar('T')

class Space(Generic[T]):
    r"""
    A minimal stateless implementation of base space used for jax
    
    """

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError
    
    def contains(self, x: Any):
        raise NotImplementedError
    



