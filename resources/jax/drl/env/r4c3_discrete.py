import math
from typing import Optional, Tuple, Union, Callable, List

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector.utils import batch_space
from scipy import interpolate

class LinearInterpolation(object):
    def __init__(self, ts, ys):
        """
        ts: increasing collections of times, such as [t0, t1, t2, ...]
        ys: values at ts, NDarray or 1-D array
        """
        
        self.interp = interpolate.interp1d(ts, ys, axis=0, kind="linear", fill_value="nearest")

    def evaluate(self, t):
        """
        evaluate at time t, t should be in ts
        """
        return self.interp(t)

class DiscreteLinearStateSpaceEnv(gym.Env):
    """
    should we implement following gym? 
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, 
        A: np.array, 
        Bu: np.array, 
        Bd:np.array, 
        C:np.array, 
        D:np.array, 
        x0:np.array,
        dist_fcn: LinearInterpolation,
        x_high:np.array, 
        x_low:np.array, 
        n_actions: Union[int, List[int]],
        u_high, 
        u_low,
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
        # asset that the shapes are correct
        assert self.A.shape[0] == self.A.shape[1]
        assert self.Bu.shape[0] == self.A.shape[0]
        assert self.Bd.shape[0] == self.A.shape[0]
        assert self.C.shape[1] == self.A.shape[0]
        assert self.D.shape[0] == self.C.shape[0]
        assert self.x0.shape[0] == self.A.shape[0]
        assert self.u_high.shape[0] == self.Bu.shape[1]
        assert self.u_low.shape[0] == self.Bu.shape[1]
        assert self.u_high.shape[0] == self.u_low.shape[0]

        # simulation 
        self.ts = ts
        self.te = te 
        self.dt = dt 
        self.t = self.ts # initial model clock

        # x high limit 
        self.high = np.array(x_high,
            dtype=np.float32,
        ) if type(x_high) is not np.ndarray else x_high.astype(np.float32)
        # x low limit
        self.low = np.array(x_low,
            dtype=np.float32,
        ) if type(x_low) is not np.ndarray else x_low.astype(np.float32)

        # disturbance
        self.dist_fcn = dist_fcn 
        
        # discrete action space
        # np.array(scalar).shape = ()
        if type(n_actions) is int:
            n_actions = [n_actions]
        
        self.n_actions = np.array(n_actions)
        
        if len(n_actions) == 1:
            self.action_space = spaces.Discrete(n_actions[0])
        # TODO: multidiscrete not tested yet
        elif len(n_actions) > 1:
            self.action_space = spaces.MultiDiscrete(n_actions)
        
        self.observation_space = self._get_observation_space()

        # reward
        self.reward = 0
        self.done = False

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
        state = np.array(state).reshape(-1,1)
        action = np.array(action).reshape(-1,1)
        disturbance = np.array(disturbance).reshape(-1,1)

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
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            control_inputs = action/(self.n_actions-1)*(self.u_high - self.u_low) + self.u_low
        else:
            logger.error(f"Action space {type(self.action_space)} is not implemented for current gym environment.")

        return control_inputs
    
    def _get_observation_space(self):
        """
        can be overwritten by high-level wrapper
        """
        return spaces.Box(self.low, self.high, dtype=np.float32)
        
    def _reward_fcn(self):
        
        return -10 if self.done else 1 

    def _is_done(self):
        
        self.done = bool(
            self.t >= self.te
        )

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        state = self.state

        # disturbance
        self._disturbance = self._get_disturbance()
        
        # advance one step
        self.state, y = self.model(state, action, self._disturbance)
        
        # update model clock
        self.t += self.dt
        
        # terminal conditions: simulation over time, state over bounds
        self._is_done()

        # reward 
        reward = self._reward_fcn()
        
        # render: NOT IMPLEMENTED

        return np.array(self.state, dtype=np.float32), reward, self.done, False, {}

    def _get_disturbance(self):
        """
        system disturbance at time t: use interpolate
        """
        return self.dist_fcn.evaluate(self.t)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.t = self.ts
        self.state = self.x0 + self.np_random.uniform(low=-0.5, high=0.5, size=np.array(self.x0).shape)

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def close(self):
        pass 


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
        weights: Optional[List[float]] = None,
        Tz_high: Optional[List[float]] = None,
        Tz_low: Optional[List[float]] = None,
        cop: Optional[float] = 3.0,
        energy_price: Optional[List[float]] = None,
        ts: Optional[int] = 0, 
        te: Optional[int] = 1,
        dt: Optional[int] = 0.1,
        n_next_steps: Optional[int] = 4,
        n_prev_steps: Optional[int] = 4,
        render_mode: Optional[str] = None):
        
        self.rc_params = rc_params
        
        # get linear state space
        self.A, self.Bu, self.Bd, self.C, self.D = self._getABCD()

        # get piece wise continuous disturbance model
        # t, d = disturbances
        dist_fcn = LinearInterpolation(*disturbances)
        
        # reward parameters
        self.cop = cop 
        self.energy_price = energy_price if energy_price else np.ones(24) 

        Tz_high_default = [30.0 for i in range(24)]
        Tz_low_default = [12.0 for i in range(24)]
        Tz_high_default[8:18] = [26.0]*(18-8)
        Tz_low_default[8:18] = [22.0]*(18-8)
        self.Tz_high = Tz_high if Tz_high else Tz_high_default
        self.Tz_low = Tz_low if Tz_low else Tz_low_default

        self.weights = weights if weights else [1.0, 1.0, 1.0]

        # observation space for DRL algorithms
        self.n_next_steps = n_next_steps 
        self.n_prev_steps = n_prev_steps

        # conditional
        self.history = {}
        if self.n_prev_steps > 0:
            self.history['Tz'] = [20.]*self.n_prev_steps
            self.history['P'] = [0.]*self.n_prev_steps

        # initialize
        super(R4C3DiscreteEnv, self).__init__(
            A = self.A,
            Bu = self.Bu,
            Bd = self.Bd,
            C = self.C,
            D = self.D,
            x0 = x0,
            dist_fcn = dist_fcn,
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
        Bu = np.zeros((3, 1))
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
        


    def _get_objective_terms(self, Tz, q_hvac, action):
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
        du = float(abs((action - self.action_prev)/(self.n_actions-1)))

        return [cost, dTz, du]

    def _reward_fcn(self, objective):
        # maximize reward: negate objective
        # we can overwrite this reward function using gym.wrapper

        energy_cost, max_T_violations, du = objective 
        w1, w2, w3 = self.weights 

        return float(-(w1*energy_cost + w2*max_T_violations + w3*du))

    def _get_observation_space(self):
        # [t, Tz, To, solar, power, price, To for next n steps, solar for next n steps, price for next n steps, Tz from previous m steps, power from previous m steps]
        high = np.array([86400.,35., 40., 4., 4., 1.0]+\
                [40.]*self.n_next_steps+[4.]*self.n_next_steps+[4.]*self.n_next_steps+\
                [35]*self.n_prev_steps+[4.]*self.n_prev_steps)
        
        low = np.array([0., 12., 0., 0., 0., 0.]+\
                [0.]*self.n_next_steps+[0.]*self.n_next_steps+[0.]*self.n_next_steps+\
                [12.]*self.n_prev_steps+[0.]*self.n_prev_steps) 

        return spaces.Box(low, high, dtype=np.float32)


    # TODO: improve the object design, typically users should not overwrite step() method, but providing customized observation, reward and termination
    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        state = self.state
        
        # get disturbance at time t
        disturbance = self._get_disturbance()

        # LSSM step
        state_next, y = self.model(state, action, disturbance)
        self.state = state_next
        self.t += self.dt

        # check termination
        # --------------------------------
        self._is_done()

        # get observation
        # --------------------------------
        # unpack output
        Tz, q_hvac = y 

        # construct DRL observations
        obs_next = self._get_observation(state_next, y, disturbance)
        self.observation = obs_next

        # get rewards
        # -----------------------------------
        objective = self._get_objective_terms(Tz, q_hvac, action)
        reward = self._reward_fcn(objective)

        # update history if needed
        # -----------------------------------
        self._update_history(action, Tz, abs(q_hvac)/self.cop)

        return np.array(obs_next, dtype=np.float32), reward, self.done, False, {}
    
    def _update_history(self, action, Tz, power):
        # update action history
        self.action_prev = action

        # update Tz history
        Tz_history = np.roll(self.history['Tz'], -1)
        Tz_history[-1] = Tz
        self.history['Tz'] = Tz_history 

        # update power history
        power_history = np.roll(self.history['P'], -1)
        power_history[-1] = power
        self.history['P'] = power_history

    def _get_observation(self, state_next, y, disturbance):
        # unpack
        Tz, Twe, Twi  = state_next
        Tz, q_hvac = y
        # disturbance: assume q_win is from solar
        # TODO: it's better to use a weather file to generate solar radiation
        To, q_int, q_win, q_rad = disturbance
        dists_next_n_steps = self._get_disturbance_next_n_steps()
        solar_next_n_steps = dists_next_n_steps[:, 2]
        To_next_n_steps = dists_next_n_steps[:, 0]

        # time
        t = self.t
        h = int(int(t)%86400/3600)

        # initialize
        obs = np.zeros(6+self.n_next_steps*3+self.n_prev_steps*2)

        # set observation
        obs[0] = h
        obs[1] = Tz
        obs[2] = To
        obs[3] = q_win
        obs[4] = abs(q_hvac)/self.cop
        obs[5] = self.energy_price[h]

        # set next steps
        if self.n_next_steps > 0:
            for i in range(self.n_next_steps):
                obs[6+i] = To_next_n_steps[i]
                obs[6+self.n_next_steps+i] = solar_next_n_steps[i]
                obs[6+self.n_next_steps*2+i] = self.energy_price[(h+i+1)%24]

        # set previous steps
        if self.n_prev_steps > 0:
            for i in range(self.n_prev_steps):
                obs[6+self.n_next_steps*3+i] = self.history['Tz'][i]
                obs[6+self.n_next_steps*3+self.n_prev_steps+i] = self.history['P'][i]

        return obs

    def _get_disturbance_next_n_steps(self):
        """
        get solar radiation for [t-n_prev_steps, t+n_next_steps]

        """
        # initialize 
        t = [self.t + self.dt*(i+1) for i in range(0, self.n_next_steps)]

        # disturbance
        disturbance_next_n_steps = self.dist_fcn.evaluate(t)

        return disturbance_next_n_steps
        

    def reset(self, 
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        ):
        # reset previous action for starting 
        self.action_prev = 0
        if self.n_prev_steps > 0:
            self.history['Tz'] = np.array([20.]*self.n_prev_steps)
            self.history['P'] = np.array([0.]*self.n_prev_steps)

        state_init, _ = super().reset(seed=seed)

        t = self.t
        h = int(int(t)%86400/3600)

        Tz, _, _ = state_init 

        #return np.hstack([state_init, 0.], dtype=np.float32), {}
        disturbance = self._get_disturbance()
        self.observation = self._get_observation(state_init, (state_init[0], 0), disturbance)
        
        print("env is reset!")
        return np.array(self.observation, dtype=np.float32), {}