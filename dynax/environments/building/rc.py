import jax
import jax.numpy as jnp
from jax import lax 
from gymnax.environments import environment, spaces
import chex
from flax import struct

@struct.dataclass
class EnvState:
    x: 