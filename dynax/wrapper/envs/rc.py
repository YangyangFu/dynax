from __future__ import annotations
from typing import Tuple, Union, Optional, TypeVar, SupportsFloat, Any, Callable

import pandas as pd 
import os 

import jax 
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.core.frozen_dict import freeze
from flax.struct import dataclass, field

from dynax.models.RC import Continuous4R3C
from dynax.simulators.simulator import DifferentiableSimulator as DS
from dynax.wrapper.core import Env
from dynax.wrapper.spaces import Discrete, Box
from dynax.agents.tabular import Tabular

Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey
Parameter = flax.core.FrozenDict

# default parameters for RC model
# Cai, Cwe, Cwi, Re, Ri, Rw, Rg
params = [10384.31640625, 499089.09375, 1321535.125,
        1.5348844528198242, 0.5000327825546265, 1.000040054321289, 
        20.119935989379883]
rc = {'Cai': params[0], 'Cwe': params[1], 'Cwi': params[2],
    'Re': params[3], 'Ri': params[4], 'Rw': params[5], 'Rg': params[6]
    }
PARAMS = freeze({'params': {'simulator':{'model': rc}}})
T_LOW = [12. for i in range(24)]
T_LOW[8:18] = [22.] * (18-8)
T_HIGH = [30. for i in range(24)]
T_HIGH[8:18] = [26.] * (18-8)

PRICE = [0.02987, 0.02987, 0.02987, 0.02987,
        0.02987, 0.02987, 0.04667, 0.04667,
        0.04667, 0.04667, 0.04667, 0.04667,
        0.15877, 0.15877, 0.15877, 0.15877,
        0.15877, 0.15877, 0.15877, 0.04667,
        0.04667, 0.04667, 0.02987, 0.02987]
WEIGHTS = [100., 1.0]

# convert to jnp array to support jit
T_LOW = jnp.array(T_LOW)
T_HIGH = jnp.array(T_HIGH)
PRICE = jnp.array(PRICE)
WEIGHTS = jnp.array(WEIGHTS)

## Get Disturbance
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = './external-data'
data_file = 'disturbance_1min.csv'
data = pd.read_csv(os.path.join(file_path, data_path, data_file), index_col=[0])
START_TIME = 0.
DT = 60
index = jnp.arange(START_TIME, START_TIME+len(data)*DT, DT)
data.index = index
END_TIME = data.index[-1]

disturbance_cols = ['out_temp', 'qint_lump', 'qwin_lump', 'qradin_lump']
disturbance = data[disturbance_cols]
DISTURBANCE = Tabular(ts=index, xs=disturbance.values, mode='linear')

# CLASS DEFINITION
@dataclass
class EnvStates:
    x: jnp.ndarray # [Tout, Text_wall, Tint_wall] 
    time: float = 0.0

STATES = EnvStates(x=jnp.array([20., 30., 26.]), time=START_TIME)

class RC(Env):
    """ Differentiable RC environment with discrete actions
    """
    start_time: float = START_TIME
    end_time: float = END_TIME
    dt: float = DT
    COP: float = 2.5
    price: Array = PRICE
    T_low: Array = T_LOW
    T_high: Array = T_HIGH
    weights: Array = WEIGHTS
    u_high: float = 0.
    u_low: float = -10.
    num_actions: int = 101
    # setup disturbance model
    disturbance: Tabular = DISTURBANCE
    
    def setup(self):
        """Sets up the environment by specifying a differentiable model
        """


        # set up simulator
        model = Continuous4R3C()
        self.simulator = DS(
            model=model, 
            dt=self.dt, 
            mode_interp='linear', 
            start_time=self.start_time
        )

    def __call__(
            self, 
            action: int,
            states: EnvStates, 
        ) -> Tuple[jnp.ndarray, float, bool, bool, dict[str, Any], EnvStates]:
        """ Run one step of the environment dynamics.

        Args:
            key: random key, used for stochastic environments. Not supported yet.
            state: current state
            action: action to take
            params: parameters of the environment

        Returns:
            obs_next: next observation
            reward: reward
            terminated: whether the episode is terminated
            truncated: whether the episode is truncated
            info: additional information
            states_next: next state. JAX is stateless design, so we need to return the next state explicitly.
        """
        # convert action to control signal
        action = self._action_to_control_(action)

        # time table for disturbance
        dist = self.disturbance(states.time)
        
        # construct inputs based on model needs: 
        inputs = jnp.array([dist[0], dist[1], action, dist[2], dist[3]]).reshape(1,-1)
        states_next, outs = self.simulator(states.x, inputs)

        # update states
        states_next = EnvStates(
            x=states_next[0], # the first dimension is the time dimension 
            time=states.time+self.dt
        )

        # observation
        obs_next = self._get_obs(states_next, outs[0], dist, action)

        # reward function
        reward, cost, dTz = self._get_reward(obs_next)

        # terminated
        terminated = self._is_terminated(states_next)

        # truncated: not used. 
        # use a TimeLimit Wrapper to enable this
        # should be used with a maximum episode length in the future
        truncated = False

        # additional info
        info = {'time': states_next.time ,'cost': cost, 'dTz': dTz, 'states': states_next.x}


        return obs_next, reward, terminated, truncated, info, states_next

    def _action_to_control_(self, action: int) -> float:
        """ Convert action to control signal
        """
        # not good for jit
        #if not self.action_space.contains(action):
        #    raise ValueError(f"Invalid action {action} for {self.name} environment")
        
        action = action * (self.u_high - self.u_low) / (self.num_actions - 1) + self.u_low

        return action
    
    def _control_to_action_(self, control: float) -> int:
        """ Convert control signal to action
        """
        action = ((control - self.u_low) * (self.num_actions - 1) / (self.u_high - self.u_low)).astype(int)

        return action

    def render(self):
        """ Render the environment
        """
        pass

    @property
    def action_space(self) -> Discrete:
        """ Returns the action space of the environment
        """
        return Discrete(self.num_actions)
    
    @property
    def observation_space(self) -> Box:
        """ Returns the observation space of the environment
        """
        return self._get_observation_space()

    def _get_observation_space(self):
        # [t, Tz, To, q_int, q_sol, power]
        high = jnp.array([86400.,40., 40., 5., 5.])
        low = jnp.array([0., 12., 12., 0., 0.]) 

        return Box(low, high, dtype=jnp.float32)        

    def _get_obs(
            self, 
            states: EnvStates,
            outputs: jnp.ndarray, 
            disturbance: jnp.ndarray, 
            action:float) -> jnp.ndarray:
        """ Returns the observation of the specific environment

        Args:
            states: current state of the environment
        
        Returns:
            time: time of the day in seconds
            Tz: zone temperature in degree Celsius
            To: outdoor temperature in degree Celsius
            q_sol: solar radiation in kW/m^2
            power: power consumption in kW
        """

        return jnp.array([states.time, 
                          outputs[0], 
                          disturbance[0], 
                          disturbance[3], 
                          -action/self.COP]) # power should be positive

    def _get_reward(
            self, 
            obs: jnp.ndarray) -> float:
        """ Returns the reward of the specific environment.
        
        The reward is defined as the weighted sum of power consumption and thermal discomfort. 

        Args:
            state: current state of the environment
        """
        # get time index
        h = (obs[0].astype(int)%86400 / 3600).astype(int)

        # get power usage
        energy = obs[4] * self.dt / 3600. # kWh

        # get energy cost
        cost = self.price[h]*energy # $/kWh

        # get thermal discomfort
        dTz = nn.activation.relu(obs[1] - self.T_high[h]) + nn.activation.relu(self.T_low[h] - obs[1])

        # get reward
        reward = -self.weights[0]*cost - self.weights[1]*dTz
        
        return reward, cost, dTz

    # TODO: not jittable
    def _is_terminated(self, states: EnvStates) -> bool:
        """ Check if the episode is terminated due to states violations
        """
        #return bool(
        #    (states.x > 40.).any() 
        #    or (states.x < 12.).any()
        #)
        terminated = jnp.where((states.x > 40.).any() or (states.x < 12.).any(), 1, 0)
        
        return terminated
    
    def reset(self,
            key: PRNGKey,
            params: Parameter = PARAMS,
            states: EnvStates = STATES,
            determnistic: bool = True
        ) -> Tuple[Array, EnvStates, Parameter]:
        """ Reset the environment to initial state
        """
        # if stochastic, add uniform noise to initial states
        if not determnistic:
            states.x += jax.random.uniform(key, minval=-1, maxval=1., shape=states.x.shape)

        # reset simulator: NOT NEEDED as the simulator as an environment is already trained.
        
        # reset observation
        key, key_reset = jax.random.split(key)
        init_dist = self.disturbance.init(key, START_TIME)
        dist = self.disturbance.apply(init_dist, at = START_TIME)
        obs = self._get_obs(
            states=states, 
            outputs=jnp.array([states.x[0]]), 
            disturbance=dist, 
            action=self.num_actions-1) # set action to the lowest
        
        print(f"Environment {self.id} is reset.")

        return obs, states, params
        


        

        
        
