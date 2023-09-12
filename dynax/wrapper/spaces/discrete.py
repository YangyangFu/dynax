from typing import Any, Union

import chex
import jax
import jax.numpy as jnp 

from dynax.wrapper.spaces import Space

Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey

class Discrete(Space):

    def __init__(self,
                 n: Union[int, jnp.integer],
                 start: Union[int, jnp.integer] = 0,
                 dtype = jnp.int32
                 ):
        r"""Constructor of :class:`Discrete` space.

        This will construct the space :math:`\{\text{start}, ..., \text{start} + n - 1\}`.

        Args:
            n (int): The number of elements of this space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the ``Dict`` space.
            start (int): The smallest element of this space.
        """

        assert jnp.issubdtype(type(n), jnp.integer), f"Expects `n` to be an integer, actual dtype: {type(n)}"
        assert n > 0, "n (counts) have to be positive"
        assert jnp.issubdtype(type(start), jnp.integer), f"Expects `start` to be an integer, actual dtype: {type(start)}"

        self.dtype = dtype
        self.n = jnp.array(n).astype(self.dtype)
        self.start = jnp.array(start).astype(self.dtype)

    def sample(self, rng: PRNGKey) -> Array:
        """ Sample a random action uniformly from set of categorical choices
        """

        rint = jax.random.randint(rng, shape=(), minval=0, maxval=self.n, dtype=self.dtype)
        return self.start + rint
    
    def contains(self, x: Any) -> bool:
        """ Return boolean specifying if x is a valid member of this space
        """

        if isinstance(x, int):
            x = jnp.array(x).astype(self.dtype)
        elif isinstance(x, (jnp.generic, jnp.ndarray)) and (
            jnp.issubdtype(x.dtype, jnp.integer) and x.shape == ()
        ):
            x = jnp.array(x).astype(self.dtype)
        else:
            return False
        
        return bool(self.start <= x < self.start + self.n)
    
    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        if self.start != 0:
            return f"Discrete({self.n}, start={self.start})"
        return f"Discrete({self.n})"
    
    def __eq__(self, other: Any) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )
    
