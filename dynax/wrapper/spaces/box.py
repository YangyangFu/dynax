from typing import Any, Union, Sequence, SupportsFloat

import chex
import jax
import jax.numpy as jnp 

from dynax.wrapper.spaces import Space

Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey

def is_float_integer(var: Any) -> bool:
    """Checks if a variable is an integer or float."""
    return jnp.issubdtype(type(var), jnp.integer) or jnp.issubdtype(type(var), jnp.floating)

class Box(Space):
    r"""A (possibly unbounded) box in :math:`\mathbb{R}^n`.

    Specifically, a Box represents the Cartesian product of n closed intervals.
    Each interval has the form of one of :math:`[a, b]`, :math:`(-\infty, b]`,
    :math:`[a, \infty)`, or :math:`(-\infty, \infty)`.

    There are two common use cases:

    * Identical bound for each dimension::

        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(-1.0, 2.0, (3, 4), float32)

    * Independent bound for each dimension::

        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box([-1. -2.], [2. 4.], (2,), float32)
    """

    def __init__(self,
                 low: Union[int, float, Array],
                 high: Union[int, float, Array],
                 shape: Union[Sequence[int], None] = None,
                 dtype = jnp.float32,
    ):
        r"""Constructor of :class:`Box`.

        The argument ``low`` specifies the lower bound of each dimension and ``high`` specifies the upper bounds.
        I.e., the space that is constructed will be the product of the intervals :math:`[\text{low}[i], \text{high}[i]]`.

        If ``low`` (or ``high``) is a scalar, the lower bound (or upper bound, respectively) will be assumed to be
        this value across all dimensions.

        Args:
            low (SupportsFloat | np.ndarray): Lower bounds of the intervals. If integer, must be at least ``-2**63``.
            high (SupportsFloat | np.ndarray]): Upper bounds of the intervals. If integer, must be at most ``2**63 - 2``.
            shape (Optional[Sequence[int]]): The shape is inferred from the shape of `low` or `high` `np.ndarray`s with
                `low` and `high` scalars defaulting to a shape of (1,)
            dtype: The dtype of the elements of the space. If this is an integer type, the :class:`Box` is essentially a discrete space.

        Raises:
            ValueError: If no shape information is provided (shape is None, low is None and high is None) then a
                value error is raised 
        """

        assert (
            dtype is not None
        ), "Box dtype must be explicitly provided, cannot be None."
        self.dtype = dtype 

        # determine shape if not provided explicitly
        if shape is not None:
            assert all(
                jnp.issubdtype(type(dim), jnp.integer) for dim in shape
            ), f"Expects all shape elements to be an integer, actual type: {tuple(type(dim) for dim in shape)}"
            shape = tuple(int(dim) for dim in shape)
        elif isinstance(low, Array):
            shape = low.shape
        elif isinstance(high, Array):
            shape = high.shape
        elif is_float_integer(low) and is_float_integer(high):
            shape = (1,)
        else:
            raise ValueError(
                f"Box shape is inferred form low and high, expected their types to be jnp.ndarray, an integer or a float, actual type low: {type(low)}, high: {type(high)}."
            )

        self._shape = shape

        # boundness check
        low = jnp.full(shape, low, dtype=self.dtype) if is_float_integer(low) else low 
        self.bounded_below: Array[jnp.bool_] = -jnp.inf < low 

        high = jnp.full(shape, high, dtype=self.dtype) if is_float_integer(high) else high 
        self.bounded_above: Array[jnp.bool_] = jnp.inf > high 

        # TODO: if the boundary is out of precision bound, need add bounds to those boundaries.
        # here we assume the given low and high are bounded. we dont deal with type overflow now.
        assert (
            all(self.bounded_below)
        ), f"low is not bounded, {self.bounded_below}"

        assert(
            all(self.bounded_above)
        ), f"high is not bounded, {self.bounded_above}"

        # assert low/high shape
        assert isinstance(low, Array)
        assert (
            low.shape == shape
        ), f"low.shape doesn't match provided shape, low.shape: {low.shape}, provided shape: {shape}"

        assert isinstance(high, Array)
        assert (
            high.shape == shape
        ), f"high.shape doesn't match provided shape, high.shape: {high.shape}, provided shape: {shape}"

        # check we dont have invalid low/high
        if jnp.any(low > high):
            raise ValueError(
                f"Some low values are greater than high, low={low}, high={high}"
            )
        if jnp.any(jnp.isposinf(low)):
            raise ValueError(
                f"No low value can be equal to `jnp.inf`, low={low}"
            )
        if jnp.any(jnp.isneginf(high)):
            raise ValueError(
                f"No high value can be equal to `-jnp.inf`, high={high}"
            )        

        # precision check
        low_precision = get_precision(low.dtype)
        high_precision = get_precision(high.dtype)
        dtype_precision = get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            Warning(f"Box bound precision lowered by casting to {self.dtype}")

        self.low = low.astype(self.dtype)
        self.high = high.astype(self.dtype)

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    def sample(self, rng: PRNGKey) -> Array:
        r"""Generates a single random sample inside the Box.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Args:
            rng: A random number generator (key)

        Returns:
            A sampled value from the Box    
        """
        # assume given high is closed, for python, we need use open bracket on the right
        high = self.high if self.dtype.dtype.kind == "f" else self.high.astype(jnp.int32) + 1
        
        # low/high has to bounded
        bounded = self.bounded_below & self.bounded_above
        assert (
            all(bounded)
        ), f"provided low and high should be bounded, actual boundness {bounded}."


        if self.dtype.dtype.kind in ["i", "u", "b"]:
            sample_fcn = jax.random.randint 
        else:
            sample_fcn = jax.random.uniform
            
        sample = sample_fcn(rng, shape=bounded.shape, minval=self.low, maxval=high, dtype=self.dtype)

        return sample

    def contains(self, x: Any) -> bool:
        """Return boolean specifying if x is a valid member of this space"""
        if not isinstance(x, Array):
            Warning("Casting input x to jax array.")
            try:
                x = jnp.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False
             
        return bool(
            jnp.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and jnp.all(x >= self.low)
            and jnp.all(x <= self.high)
        )

    def __repr__(self) -> str:
        """Gives a string representation of this space."""

        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"

    
    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        return (
            isinstance(other, Box)
            and self.dtype == other.dtype
            and (self.shape == other.shape)
            # and (self.dtype == other.dtype)
            and jnp.allclose(self.low, other.low)
            and jnp.allclose(self.high, other.high)
        )

def get_precision(dtype: jnp.dtype) -> SupportsFloat:
    """Get precision of a data type."""
    if jnp.issubdtype(dtype, jnp.floating):
        return jnp.finfo(dtype).precision
    else:
        return jnp.inf