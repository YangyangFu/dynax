import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector.utils import batch_space
from scipy import interpolate


class DiscreteLinearStateSpaceEnv(gym.Env):
    """
    ## Description
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.
    ## Action Space

    ## Observation Space
    The observation space is the state space, which assumes fully-observable states in the dynamic system:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    ## Rewards
    The state space outputs are modeled as reward.

    ## Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    ## Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    ## Arguments
    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```
    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, 
        A, 
        Bu, 
        Bd, 
        C, 
        D, 
        x0,
        x_high, 
        x_low, 
        n_actions, 
        u_high, 
        u_low,
        reward_fcn: Optional[function] = None, 
        ts: Optional[int] = 0, 
        te: Optional[int] = 1,
        dt: Optional[int] = 0.1,
        render_mode: Optional[str] = None):
    
        # assign
        self.A = A
        self.Bu = Bu
        self.Bd = Bd
        self.C = C 
        self.D = D
        self.x0 = x0 # initial state
        self.integrator = "euler"
        self.u_high = np.array(u_high)
        self.u_low = np.array(u_low)

        # reward function
        self.reward_fcn = reward_fcn        

        # simulation 
        self.ts = ts
        self.te = te 
        self.dt = dt 
        self.t = self.ts # initial model clock

        # x high limit 
        high = np.array(x_high,
            dtype=np.float32,
        ) if type(x_high) is not np.ndarray else x_high.astype(np.float32)
        # x low limit
        low = np.array(x_low,
            dtype=np.float32,
        ) if type(x_low) is not np.ndarray else x_low.astype(np.float32)

        # disturbance
        self.disturbance = None 

        # discrete action space
        # single action
        n_actions = np.array(n_actions)
        assert len(n_actions) > 0, "Number of discrete actions cannot be 0 !"
        if len(n_actions) == 1:
            self.action_space = spaces.Discrete(n_actions[0])
        # TODO: multidiscrete not tested yet
        elif len(n_actions) > 1:
            self.action_space = spaces.MultiDiscrete(n_actions)
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # reward
        self.reward = 0

        # skpi render mode
        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

        # assertions: dimensions should match

    def _linear_state_space(self, state, action, disturbance):
        state = np.array(state)
        xdot = self.A @ state + self.Bu @ action + self.Bd @ disturbance
        y = self.C @ state + self.D @ action

        # solve with integrator
        if self.integrator == "euler":
            state_next = state + self.dt * xdot
        else:
            logger.error(f"integrator {self.integrator} is not implemented. ")

        return (state_next, y)
        
    def model(self, state, action, disturbance):
        # map discrete action to control inputs
        action = self._action_to_control(action)

        return self._linear_state_space(state, action, disturbance)
        
    def _action_to_control(self, action):
        if isinstance(self.action_space, gymnasium.spaces.discrete.Discrete):
            control_inputs = action/(self.n_actions-1)*(self.u_high - self.u_low) + self.u_low
        else:
            logger.error(f"Action space {type(self.action_space)} is not implemented for current gym environment.")

        return control_inputs

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        state = self.state

        # advance one step
        self.state, reward = self.model(state, action, self.disturbance)
        # update model clock
        self.t += self.dt
        
        # terminal conditions: simulation over time, state over bounds
        terminated = bool(
            self.t >= self.te
        )

        # render: NOT IMPLEMENTED

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        # initial state + random noise  
        self.t = self.ts
        self.state = self.x0 + self.np_random.uniform(low=low, high=high, size=np.array(self.x0).shape)

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def close(self):
        pass 

class LinearInterpolation(object):
    def __init__(self, ts, ys):
        """
        ts: increasing collections of times, such as [t0, t1, t2, ...]
        ys: values at ts, NDarray or 1-D array
        """
        if len(np.array(ys).shape) == 1:
            self.interp = interpolate.interp1d(ts, ys)
        elif len(np.array(ys).shape) > 1:
            self.interp = interpolate.LinearNDInterpolator(ts, ys)
        
    def evaluate(self, t):
        """
        evaluate at time t, t should be in ts
        """
        return self.interp(t)

class R4C3DiscreteEnv(DiscreteLinearStateSpaceEnv):
    """
    RC state space model with discrete control actions

    x = [Tz, Twe, Twi]
    u = q_hvac
    d = [Toa, q_conv_i, q_sol_e, q_rad_i]
    y = [Tz, q_hvac]

    ** Equation
    xdot = A*x + Bu*u + Bd*d
    y = C*x + D*u
    """
    def __init__(self, 
        rc_params,
        x0,
        x_high, 
        x_low, 
        n_actions, 
        u_high, 
        u_low, 
        disturbances,
        weights: Optional[list[float]] = None,
        Tz_high: Optional[list[float]] = None,
        Tz_low: Optional[list[float]] = None,
        cop: Optional[float] = 3.0,
        energy_price: Optional[list[float]] = None,
        ts: Optional[int] = 0, 
        te: Optional[int] = 1,
        dt: Optional[int] = 0.1,
        render_mode: Optional[str] = None):
        
        self.Cai, self.Cwe, self.Cwi, self.Re, self.Ri, self.Rw, self.Rg = rc_params
        
        # get linear state space
        self.A, self.Bu, self.Bd, self.C, self.D = self._getABCD()

        # get piece wise continuous disturbance model
        # t, d = disturbances
        self.dist = LinearInterpolation(*disturbances)
        
        # reward parameters
        self.cop = cop 
        self.energy_price = energy_price if not energy_price else np.ones(24) 

        Tz_high_default = [30.0 for i in range(24)]
        Tz_low_default = [12.0 for i in range(24)]
        Tz_high_default[8:18] = [26.0]*(18-8)
        Tz_low_default[8:18] = [22.0]*(18-8)
        self.Tz_high = Tz_high if Tz_high else Tz_high_default
        self.Tz_low = Tz_low if Tz_low else Tz_low_default

        self.weights = weights if weights else [1.0, 1.0, 1.0]

        # initialize
        super(R4C3DiscreteEnv, self).__init__(
            A = self.A,
            Bu = self.Bu,
            Bd = self.Bd,
            C = self.C,
            D = self.D,
            x0 = x0,
            x_high = x_high, 
            x_low = x_low, 
            n_actions = n_actions, 
            u_high = u_high, 
            u_low = u_low, 
            ts = ts, 
            te = te, 
            dt = dt,
            render_mode = render_mode
            )

    def _getABCD(self):
        # unpack
        Cai, Cwe, Cwi, Re, Ri, Rw, Rg = self.rc_params
        # initialzie
        A = np.zeros((3, 3))
        Bu = np.zeros((1, 1))
        Bd = np.zeros((3, 4))
        C = np.zeros((2, 3))
        D = np.zeros((2, 1))

        # set matrix
        A[0, 0] = -1/Cai*(1/Rg+1/Ri)
        A[0, 2] = 1/(Cai*Ri)
        A[1, 1] = -1/Cwe*(1/Re+1/Rw)
        A[1, 2] = 1/(Cwe*Rw)
        A[2, 0] = 1/(Cwi*Ri)
        A[2, 1] = 1/(Cwi*Rw)
        A[2, 2] = -1/Cwi*(1/Rw+1/Ri)
        
        Bu[0, 0] = 1/Cai

        Bd[0, 0] = 1/(Cai*Rg)
        Bd[0, 1] = 1/Cai
        Bd[1, 0] = 1/(Cwe*Re)
        Bd[1, 2] = 1/Cwe
        Bd[2, 3] = 1/Cwi

        C[0, 0] = 1

        D[1, 0] = 1

        return A, Bu, Bd, C, D
        
    def _get_disturbance(self):
        """
        system disturbance at time t: use interpolate
        """
        return self.dist(self.t)

    def get_objective_terms(self, Tz, q_hvac, action):
        """
        q_hvac: control input: negative for cooling
        action: discrete action for gym
        """
        # energy cost
        h = int(int(self.t)%86400/3600)
        cost = abs(q_hvac)/self.cop*self.energy_price[h]/self.dt/3600

        # reference violation
        dTz = max(Tz - self.Tz_high[h], self.Tz_low[h] - Tz, 0)
        
        # control slew
        du = abs((action - self.action_prev)/(self.n_actions-1))

        return [cost, dTz, du]

    def reward_fcn(self, objective):
        # maximize reward: negate objective
        # we can overwrite this reward function using gym.wrapper

        energy_cost, max_T_violations, du = objective 
        w1, w2, w3 = self.weights 

        return -(w1*energy_cost + w2*max_T_violations + w3*du)

    def step(self, action):
        # get disturbance
        self.disturbance = self._get_disturbance()
        
        # LSSM step
        state_next, y, terminated, _, _ = super(R4C3DiscreteEnv, self).step(action)

        # unpack output
        Tz, q_hvac = y 

        # get rewards
        objective = self.get_objective_terms(Tz, q_hvac, action)
        reward = self.reward_fcn(objective)

        # update history if needed
        self.action_prev = action 

        return state_next, reward, terminated, False, {}

    def reset(self, 
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        ):
        # reset previous action for starting 
        self.action_prev = 0

        return super(R4C3DiscreteEnv, self).reset(seed, options)

