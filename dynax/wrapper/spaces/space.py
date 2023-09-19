from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp 

T = TypeVar('T')
Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey

class Space(Generic[T]):
    r"""
    A minimal stateless implementation of base space used for jax
    
    """

    def sample(self, rng: PRNGKey) -> Array:
        raise NotImplementedError
    
    def contains(self, x: Any):
        raise NotImplementedError
    



