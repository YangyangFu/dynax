from abc import ABC, abstractmethod
from typing import Optional, Union

import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.ndimage import map_coordinates

class AbstractInterpolation(nn.Module, ABC):
    """ Base class for time table interpolation methods"""
    ts:jnp.ndarray 
    xs:jnp.ndarray

    def setup(self):
        assert len(self.ts.shape) == 1, "rank of ts has to be 1 !!!"
        assert len(self.xs.shape) == 2, "rank of xs has to be 2 !!!"

    @abstractmethod
    def __call__(self, at: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @property
    def order(self) -> int:
        """ Order of the interpolation method"""
        return self._order

class PiecewiseConstantInterpolation(AbstractInterpolation):
    """ Piecewise constant interpolation of the control actions """

    def order(self) -> int:
        return 0
    
    def __call__(self, at:jnp.ndarray) -> jnp.ndarray:
        """ Interpolate the values of x at time t using piecewise constant interpolation
        Args:
            ts: time values, shape (n,)
            xs: state values, shape (n, m)
            t: time to interpolate at
        Returns:
            x: interpolated state values
        """
        # Calculate the indices for the new_x values
        indices = jnp.interp(at, self.ts, jnp.arange(len(self.ts)))
        
        # Use linear interpolation to map the array onto the new coordinates
        result = map_coordinates(self.xs, [indices, jnp.arange(self.xs.shape[1])], order=self.order())
        return result 

class LinearInterpolation(AbstractInterpolation):

    def order(self) -> int:
        return 1

    def __call__(self, at: jnp.ndarray) -> jnp.ndarray:
        """ Interpolate the values of x at time t using linear interpolation
        Args:
            t: time to interpolate at
        Returns:
            x: interpolated state values
        """
        # Calculate the indices for the new_x values

        # interpolate on x
        indices = jnp.interp(at, self.ts, jnp.arange(len(self.ts)))

        # Use linear interpolation to map the array onto the new coordinates
        result = map_coordinates(self.xs, [indices, jnp.arange(self.xs.shape[1])], order=self.order())
        
        return result

class ThirdOrderHermitePolynomialInterpolation(AbstractInterpolation):

    def order(self) -> int:
        return 3

    def __call__(self, at: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Third-order Hermite polynomial interpolation is not yet implemented")


class FourthOrderPolynomialInterpolation(AbstractInterpolation):

    def order(self) -> int:
        return 4

    def __call__(self, at: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Fourth-order polynomial interpolation is not yet implemented")


if __name__ == "__main__":
    from jax import jvp, jit, jacrev, jacfwd
        
    # Define ts and xs values
    ts = jnp.array([0.0, 1.0, 2.0, 3.0])
    xs = jnp.array([[0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 6.0, 12.0]]).T
    
    # Define new x values to interpolate at
    t = jnp.array([0.5])
    
    # create a constant interpolation
    const = PiecewiseConstantInterpolation(ts, xs)
    x = const(jnp.array([0.5]))
    print("constant interpolation:", x.shape, x)
    
    # calculate gradients
    grad_fcn = jacfwd(lambda t: const(t))
    x_grad = grad_fcn(t)
    print("constant interpolation grads:", x_grad.shape, x_grad)
    
    # Create an instance of LinearInterpolation
    linear = LinearInterpolation(ts, xs)
    x = linear(t)
    
    print("linear interpolation:", x.shape, x)

    # Test grads and values
    grad_fcn = jacfwd(lambda t: linear(t))
    x_grad = grad_fcn(t)
    print("linear interpolation grads:", x_grad.shape, x_grad)
