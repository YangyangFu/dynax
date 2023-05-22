from typing import Callable, List, Tuple, Union
#from jaxtyping import Array, Float, Pytree
import jax.numpy as jnp
import flax.nn as nn

class BaseBlockStateSpace(nn):
    pass

class ContinuousLTIStateSpace(nn):
    r"""
    Continuouse state space system
    
    This model is defined as follows:

    $$\dot x_t = Ax_t + B_uu_t + B_dd_t$$
    $$ y_t = Cx_t + D_uu_t + D_dd_t$$
    
    where 

    * $x_t \in \mathbb{R}^{n_x}$ is the latent state vector at time $t$
    * $u_t \in \mathbb{R}^{n_u}$ is the control vector at time $t$
    * $d_t \in \mathbb{R}^{n_d}$ is the disturbance vector at time $t$
    * $y_t \in \mathbb{R}^{n_y}$ is the observation vector at time $t$
    * $A \in \mathbb{R}^{n_x \times n_x}$ is the trasition matrix
    * $B_u \in \mathbb{R}^{n_x \times n_u}$ is the control_to_state matrix
    * $B_d \in \mathbb{R}^{n_x \times n_d}$ is the disturbance_to_state matrix
    * $C \in \mathbb{R}^{n_y \times n_x}$ is the observation or emission matrix
    * $D \in \mathbb{R}^{n_y \times n_u}$ is the control_to_observation matrix

    Parameters
    ----------
    state_dim : int
        Dimension of the latent state vector
    control_dim : int
        Dimension of the control vector
    disturbance_dim : int
        Dimension of the disturbance vector
    observation_dim : int
        Dimension of the observation vector
    A : Array
        Transition matrix
    B_u : Array
        Control_to_state matrix
    B_d : Array
        Disturbance_to_state matrix
    C : Array
        Observation or emission matrix
    D : Array   
        Control_to_observation matrix    
    """

    state_dim: int 
    control_dim: int 
    disturbance_dim: int
    observation_dim: int
    
    initializer: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x, u, d):

        A = self.param('A',
                        self.initializer, # initialization
                        (self.state_dim, self.state_dim)) # shape
        Bu = self.param('Bu',
                        self.initializer, # initialization
                        (self.state_dim, self.control_dim))
        Bd = self.param('Bd',
                        self.initializer,
                        (self.state_dim, self.disturbance_dim))
        C = self.param('C',
                        self.initializer,
                        (self.observation_dim, self.state_dim))
        Du = self.param("Du",
                        self.initializer,
                        (self.observation_dim, self.control_dim))

        Dd = self.param("Dd",
                        self.initializer,
                        (self.observation_dim, self.disturbance_dim))
        xdot = jnp.dot(x, A.T) + jnp.dot(u, Bu.T) + jnp.dot(d, Bd.T)

        #x_next = xdot * self.dt + x
        y = jnp.dot(x, C.T) + jnp.dot(u, Du.T) + jnp.dot(d, Dd.T)
        
        return xdot, y

