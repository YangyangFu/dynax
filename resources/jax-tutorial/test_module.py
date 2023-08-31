# check how Flax stack layers of linen.Module
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d


class AbstractInterpolation(nn.Module, ABC):
    """ Base class for all interpolation methods"""

    @abstractmethod
    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray, new_x: jnp.ndarray) -> jnp.ndarray:
        pass
    
    @property
    def order(self) -> int:
        """ Order of the interpolation method"""
        return self._order
# define a nn module        
class LinearInterpolation(AbstractInterpolation):
    
    def order(self) -> int:
        return 1

    #@nn.compact
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


# call like neural network layer
class Wrapper1(nn.Module):

    def setup(self):
        self.interp = LinearInterpolation()
    
    def __call__(self, ts, xs, t):
        return self.interp.evaluate(ts, xs, t)


ts = jnp.array([0.0, 2.0, 3.0, 4.0])
xs = jnp.array([[0.0, 1.0, 4.0, 9.0], [0.0, 2.0, 6.0, 12.0]]).T
t = jnp.array([1.5, 2.5, 3.5]).reshape(-1,1)

seed = jax.random.PRNGKey(0)

wrap1 = Wrapper1()
inits = wrap1.init(seed, ts, xs, t)
res1 = wrap1.apply(inits, ts, xs, t)
print(inits)
print(res1)


# pass nn.module as an argument
class Wrapper2(nn.Module):

    interp: nn.Module

    def __call__(self, ts, xs, t):
        res = self.interp.evaluate(ts, xs, t)
        return res

wrap2 = Wrapper2(interp = LinearInterpolation())
inits = wrap2.init(seed, ts, xs, t)
res2 = wrap2.apply(inits, ts, xs, t)
print(inits)
print(res2)

# the following code will not work if the decorator @nn.compact is added to wrapper2 __call__() method
wrap2 = Wrapper2(interp = LinearInterpolation())
res2 = wrap2(ts, xs, t)
print(res2)

# class wrapper 3
class Wrapper3(nn.Module):
    mode:str 

    
    def setup(self):
        self.linear = LinearInterpolation()
        self.const = LinearInterpolation()
    
    def __call__(self, ts, xs, t):
        if self.mode=="linear":
            interp = self.linear
        elif self.mode == "constant":
            interp = self.const 
        
        return interp.evaluate(ts, xs, t)

wrap3 = Wrapper3(mode="linear")
inits = wrap3.init(seed, ts, xs, t)
res3 = wrap3.apply(inits, ts, xs, t)
print(res3)


# class wrapper 3
class Wrapper4(nn.Module):
    mode:str 

    @nn.compact
    def __call__(self, ts, xs, t):
        if self.mode=="linear":
            interp = LinearInterpolation()
        elif self.mode == "constant":
            interp = LinearInterpolation()
        
        return interp.evaluate(ts, xs, t)

wrap4 = Wrapper4(mode="linear")
inits = wrap4.init(seed, ts, xs, t)
res4 = wrap4.apply(inits, ts, xs, t)
print(res4)


# class wrapper 5
class Wrappe5(nn.Module):

    mode: str

    @nn.compact
    def __call__(self, ts, xs, t):
        if self.mode=="linear":
            interp = LinearInterpolation()
        elif self.mode == "constant":
            interp = LinearInterpolation()
        
        for i in range(10):
            res = interp.evaluate(ts, xs, t)

        return res

wrap5 = Wrappe5(mode='linear')
inits = wrap5.init(seed, ts, xs, t)
print(inits)
res5 = wrap5.apply(inits, ts, xs, t)
print(res5)


class Wrapper6(nn.Module):
    mode: str

    @nn.compact
    def __call__(self, ts, xs, t):
        if self.mode=="linear":
            interp = LinearInterpolation()
        elif self.mode == "constant":
            interp = LinearInterpolation()
        
        def roll(carry, tmp):
            i, res = carry

            res = interp.evaluate(ts, xs, t)

            return (i+1, res), res

        tmp = jnp.arange(10)
        carry_init = (tmp[0], jnp.zeros((3,2)))
        
        (_, res), _ = jax.lax.scan(roll, carry_init, tmp)

        return res

wrap6 = Wrapper6(mode='linear')
inits = wrap6.init(seed, ts, xs, t)
print(inits)
res6 = wrap6.apply(inits, ts, xs, t)
print(res6)    