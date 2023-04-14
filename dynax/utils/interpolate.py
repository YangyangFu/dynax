from abc import ABC, abstractmethod
from typing import Optional, Union

import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.ndimage import map_coordinates

class AbstractInterpolation(nn.Module, ABC):
    """ Base class for all interpolation methods"""

    @abstractmethod
    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray, new_x: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @property
    def order(self) -> int:
        """ Order of the interpolation method"""
        return self._order


class LinearInterpolation(AbstractInterpolation):
    
    def order(self) -> int:
        return 1

    def evaluate(self, ts: jnp.ndarray, xs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """ Interpolate the values of x at time t using linear interpolation
        Args:
            ts: time values
            xs: state values
            t: time to interpolate at
        Returns:
            x: interpolated state values
        """
        # Calculate the indices for the new_x values
        indices = ((t - ts[0]) / (ts[1] - ts[0])).reshape((-1, 1))
        # Use linear interpolation to map the array onto the new coordinates
        result = map_coordinates(xs, [indices, jnp.arange(xs.shape[1])], order=1)
        
        return result

class ThirdOrderHermitePolynomialInterpolation(AbstractInterpolation):

    def order(self) -> int:
        return 3

    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray, new_x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Third-order Hermite polynomial interpolation is not yet implemented")


class FourthOrderPolynomialInterpolation(AbstractInterpolation):

    def order(self) -> int:
        return 4

    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray, new_x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Fourth-order polynomial interpolation is not yet implemented")


if __file__ == "interpolate.py":

    # Define ts and xs values
    ts = jnp.array([0.0, 1.0, 2.0, 3.0])
    xs = jnp.array([[0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 6.0, 12.0]]).T
    
    # Create an instance of LinearInterpolation
    interp = LinearInterpolation()

    # Define new x values to interpolate at
    t = jnp.array([0.5])

    # Call interpolate() to get the interpolated y values
    x = interp.evaluate(ts, xs, t)
    
    print(x.shape, x)

    # Test grads and values
    from jax import jvp, jit, jacrev, jacfwd

    grad_fcn = jacfwd(lambda t: interp.evaluate(ts, xs, t))
    x_grad = grad_fcn(t)
    print(x_grad.shape, x_grad)
