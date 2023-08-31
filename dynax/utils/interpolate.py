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

class PiecewiseConstantInterpolation(AbstractInterpolation):
    """ Piecewise constant interpolation of the control actions """
    
    def order(self) -> int:
        return 0
    
    def evaluate(self, ts:jnp.ndarray, xs:jnp.ndarray, t:jnp.ndarray) -> jnp.ndarray:
        """ Interpolate the values of x at time t using piecewise constant interpolation
        Args:
            ts: time values, shape (n,)
            xs: state values, shape (n, m)
            t: time to interpolate at
        Returns:
            x: interpolated state values
        """
        result = xs[-1,:]  # Start with the last value as default
        for t_b, x_b in zip(ts[::-1], xs[::-1]):
            result = jnp.where(t < t_b, x_b, result)
        return result
        
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
        indices = jnp.interp(t, ts, jnp.arange(len(ts)))
        
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


if __name__ == "__main__":
    from jax import jvp, jit, jacrev, jacfwd
        
    # Define ts and xs values
    ts = jnp.array([0.0, 1.0, 2.0, 3.0])
    xs = jnp.array([[0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 6.0, 12.0]]).T
    
    # Define new x values to interpolate at
    t = jnp.array([0.5])
    
    # create a constant interpolation
    const = PiecewiseConstantInterpolation()
    x = const.evaluate(ts, xs, jnp.array([0.5]))
    print("constant interpolation:", x.shape, x)
    
    # calculate gradients
    grad_fcn = jacfwd(lambda t: const.evaluate(ts, xs, t))
    x_grad = grad_fcn(t)
    print("constant interpolation grads:", x_grad.shape, x_grad)
    
    # Create an instance of LinearInterpolation
    linear = LinearInterpolation()
    x = linear.evaluate(ts, xs, t)
    
    print("linear interpolation:", x.shape, x)

    # Test grads and values
    grad_fcn = jacfwd(lambda t: linear.evaluate(ts, xs, t))
    x_grad = grad_fcn(t)
    print("linear interpolation grads:", x_grad.shape, x_grad)
