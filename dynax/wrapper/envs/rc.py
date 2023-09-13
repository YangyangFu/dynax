from __future__ import annotations
from typing import Tuple, Union, Optional, TypeVar, SupportsFloat, Any, Callable
\
import pandas as pd 
import os 

import jax 
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.core.frozen_dict import freeze
from dynax.models.RC import Continuous4R3C
from dynax.simulators.simulator import DifferentiableSimulator as DS
from dynax.wrapper.core import Env
from dynax.wrapper.spaces import Discrete, Box
from dynax.agents.tabular import Tabular

# default parameters for RC model
# Cai, Cwe, Cwi, Re, Ri, Rw, Rg
params = [10384.31640625, 499089.09375, 1321535.125,
        1.5348844528198242, 0.5000327825546265, 1.000040054321289, 
        20.119935989379883]
rc = {'Cai': params[0], 'Cwe': params[1], 'Cwi': params[2],
    'Re': params[3], 'Ri': params[4], 'Rw': params[5], 'Rg': params[6]
    }
PARAMS = freeze({'params': {'simulator':{'model': rc}}})
T_LOW = [30. for i in range(24)]
T_LOW[8:18] = [22.] * (18-8)
T_HIGH = [12. for i in range(24)]
T_HIGH[8:18] = [26.] * (18-8)
PRICE = [0.02987, 0.02987, 0.02987, 0.02987,
        0.02987, 0.02987, 0.04667, 0.04667,
        0.04667, 0.04667, 0.04667, 0.04667,
        0.15877, 0.15877, 0.15877, 0.15877,
        0.15877, 0.15877, 0.15877, 0.04667,
        0.04667, 0.04667, 0.02987, 0.02987]

## Get Disturbance
data_path = './external-data'
data_file = 'disturbance_1min.csv'
data = pd.read_csv(os.path.join(data_path, data_file), index_col=[0])
ts = 0.
index = range(ts, ts+len(data)*60., 60.)
data.index = index

disturbance_cols = ['out_temp', 'qint_lump', 'qwin_lump', 'qradin_lump']
disturbance = data[disturbance_cols]
DISTURBANCE = Tabular(ts=index, xs=disturbance.values, interp='linear')

# CLASS DEFINITION
@flax.struct.dataclass
class EnvStates:
    x: jnp.ndarray # [Tout, Text_wall, Tint_wall] 
    time: float = 0.0


class RC(Env):
    """ Differentiable RC environment with discrete actions
    """
    disturbance: Tabular = DISTURBANCE
    COP: float = 2.5
    price: list[float] = PRICE
    T_low: list[float] = T_LOW
    T_high: list[float] = T_HIGH
    weights: list[float] = [100., 1.]

    def setup(self):
        """Sets up the environment by specifying a differentiable model
        """
        self.init_states = EnvStates(x=jnp.array([20., 30., 26.]), time=self.start_time)
        model = Continuous4R3C()
        self.simulator = DS(model=model, dt=self.dt, mode_interp='linear', start_time=self.start_time)

    def __call__(
            self, 
            states: EnvStates, 
            action: int,
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
        # time table for disturbance
        dist = self.disturbance(states.time)
        
        # construct inputs based on model needs: 
        inputs = jnp.array([dist[0], dist[1], action, dist[2], dist[3]]).reshape(1,-1)
        states_next, outs = self.simulator(states.x, inputs)

        # observation
        obs_next = self._get_observation(states_next, outs, dist, action)

        # reward function
        reward, cost, dTz = self._get_reward(obs_next)

        # terminated
        terminated = self._is_terminated(states_next)

        # truncated: not used. should be used with a maximum episode length in the future
        truncated = self._is_truncated(states_next)

        # additional info
        info = {'time': states.time ,'cost': cost, 'dTz': dTz}

        # update states
        states_next = EnvStates(
            x=states_next, 
            time=states.time+self.dt
        )

        return obs_next, reward, terminated, truncated, info, states_next

    def _action_to_control_(self, action: int) -> float:
        """ Convert action to control signal
        """
        return action

    def render(self):
        """ Render the environment
        """
        pass

    @property
    def name(self) -> str:
        return "RC-v1"

    @property
    def num_actions(self) -> int:
        return 11
    
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

    def _get_observation(
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
                          action/self.COP])

    def _get_reward(
            self, 
            obs: jnp.ndarray) -> float:
        """ Returns the reward of the specific environment.
        
        The reward is defined as the weighted sum of power consumption and thermal discomfort. 

        Args:
            state: current state of the environment
        """
        # get time index
        h = int(int(obs[0])%86400 / 3600)

        # get power usage
        energy = obs[4] * self.dt / 3600. # kWh

        # get energy cost
        cost = self.price[h]*energy # $/kWh

        # get thermal discomfort
        dTz = nn.activation.relu(obs[1] - self.T_high[h]) + nn.activation.relu(self.T_low[h] - obs[1])

        # get reward
        reward = -self.weights[0]*cost - self.weights[1]*dTz
        
        return reward, cost, dTz

    def _is_terminated(self, states: EnvStates) -> bool:
        """ Check if the episode is terminated due to states viOlations
        """
        return bool(
            states.x[0] > 40. 
            or states.x[0] < 12.
        )
    
    def _is_truncated(self, states: EnvStates) -> bool:
        """ Check if the episode is truncated due to time limit
        """
        return bool(
            states.time > self.end_time
        )
    
        

        
        
