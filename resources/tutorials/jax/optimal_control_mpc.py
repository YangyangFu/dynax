import jax
import jax.numpy as jnp
from jax import jit
from jax import grad
from diffrax import diffeqsolve, ODETerm, Euler, Dopri5, SaveAt, PIDController
import optax 
import pandas as pd 
import numpy as np 

class MPC():
    def __init__(self, PH, CH, dt, predictor):
        self.PH = PH
        self.CH = CH
        self.dt = dt

        # reduced order model
        self.predictor = predictor

        # building parameters
        self.occ_start = 8
        self.occ_end = 18

        # control variable bounds
        self.u_lb = -10*jnp.ones(self.PH, 1)
        self.u_ub = 0*jnp.ones(self.PH, 1)

        # optimization settings
        # ode solver
        self.ode_solver = Euler()
        
        # zone temperature control boulds
        self.T_ub = np.array([30.0 for i in range(24)])
        self.T_ub[self.occ_start:self.occ_end] = 26.0
        self.T_ub = jnp.array(self.T_ub)

        self.T_lb = np.array([12.0 for i in range(24)])
        self.T_lb[self.occ_start:self.occ_end] = 22.0
        self.T_lb = jnp.array(self.T_lb)

        # energy price
        self.price = [0.02987, 0.02987, 0.02987, 0.02987, 
                    0.02987, 0.02987, 0.04667, 0.04667, 
                    0.04667, 0.04667, 0.04667, 0.04667, 
                    0.15877, 0.15877, 0.15877, 0.15877,
                    0.15877, 0.15877, 0.15877, 0.04667, 
                    0.04667, 0.04667, 0.02987, 0.02987]

    # set methods 
    def set_time(self, time):
        self.time = time
    
    def set_state(self, state):
        self.state = state
    
    # main method
    def step(self, disturbance):
        """
        step forward in MPC

        :param state: initial state for state-space model
        :type state: jny.numpy (n,)
        :param ts: start time
        :type ts: float
        :param te: end time
        :type te: float
        :param dt: time step
        :type dt: float
        :param disturbance: disturbance for PH
        :type disturbance: jnp.numpy (PH, n)
        :return: 
        :rtype: _type_
        """
        # clock
        ts = self.time
        te = ts + self.PH * self.dt

        # model state
        state = self.state

        # get bounds
        h_ph = []
        for ph in range(self.PH):
            t = int(ts + ph*self.dt)
            h = int((t % 86400) /3600) # hour index: 0-based
            h_ph.append(h)
        
        price_ph = self.price[h_ph].reshape(-1,1)
        T_ub_ph = self.T_ub[h_ph].reshape(-1, 1)
        T_lb_ph = self.T_lb[h_ph].reshape(-1, 1)

        # call optimizer to minimize loss
        lr = 0.1
        n_epochs = 10000
        optimizer = optax.adam(learning_rate = lr)
        u0 = jnp.zeros(self.PH, 1)
        params = {'u': u0}
        opt_state = optimizer.init(params)
        for epoch in range(n_epochs):
            loss, grads = jax.value_and_grad(self.loss_mpc)(
                params, state, ts, te, self.dt, disturbance, T_ub_ph, T_lb_ph, price_ph)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_update(params, updates)

            if epoch % 10 == 0:
                print(f'epoch {epoch}, training loss: {loss}')
        
        u_ph = params['u']
        return u_ph

        # control variable is qhvac
    def loss_mpc(self, params, state, ts, te, dt, disturbance, T_ub, T_lb, price):
        u = params['u']
        d = jnp.concatenate((disturbance[:,:2], u, disturbance[:, 3:]), axis=1)
        T_vios = self.get_T_violations(state, ts, te, dt, d, T_ub, T_lb)
        u_vios = self.get_u_violations(u)

        obj = price*u*self.dt/3600 + T_vios + u_vios 
        
        return jnp.sum(obj)

    def get_T_violations(self, state, ts, te, dt, disturbance, T_ub_ph, T_lb_ph):
        # time 
        time= self.time
        # get zone model
        Cai, Cwe, Cwi, Re, Ri, Rw, Rg = self.predictor['zone_model']
        A, B, C, D = get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg)
        args = (A, B, disturbance)

        # solve zone state-space model
        ts, xs = forward(f, ts, te, dt, state, self.ode_solver, args)

        # zone temperature violations
        Tz_ph = xs[:,0]
        T_vio_ub_ph = jax.nn.relu(Tz_ph - T_ub_ph)
        T_vio_lb_ph = jax.nn.relu(T_lb_ph - Tz_ph)
        T_vios = T_vio_ub_ph.sum() + T_vio_lb_ph.sum()

        return T_vios

    def get_u_violations(self, u):
        """
        Bound control inputs to lower and upper limits
        """
        return jnp.sum(jax.nn.relu(u - self.u_ub) + jax.nn.relu(self.u_lb - u))


@jit
def get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg):
    A = jnp.zeros((3, 3))
    B = jnp.zeros((3, 5))
    C = jnp.zeros((1, 3))
    A = A.at[0, 0].set(-1/Cai*(1/Rg+1/Ri))
    A = A.at[0, 2].set(1/(Cai*Ri))
    A = A.at[1, 1].set(-1/Cwe*(1/Re+1/Rw))
    A = A.at[1, 2].set(1/(Cwe*Rw))
    A = A.at[2, 0].set(1/(Cwi*Ri))
    A = A.at[2, 1].set(1/(Cwi*Rw))
    A = A.at[2, 2].set(-1/Cwi*(1/Rw+1/Ri))

    B = B.at[0, 0].set(1/(Cai*Rg))
    B = B.at[0, 1].set(1/Cai)
    B = B.at[0, 2].set(1/Cai)
    B = B.at[1, 0].set(1/(Cwe*Re))
    B = B.at[1, 3].set(1/Cwe)
    B = B.at[2, 4].set(1/Cwi)

    C = C.at[0, 0].set(1)

    D = 0

    return A, B, C, D

@jit
def zone_state_space(t, x, A, B, d):
    x = x.reshape(-1, 1)
    d = d.reshape(-1, 1)
    dx = jnp.matmul(A, x) + jnp.matmul(B, d)
    dx = dx.reshape(-1)

    return dx

# zone model wrapper
def f(t, x, args): return zone_state_space(t, x, *args)  # args[0], args[1], args[2])

# Using for loop to update the disturbance every time step
def forward(func, ts, te, dt, x0, solver, args):
    # unpack args
    A, B, d = args

    # ode formulation
    term = ODETerm(func)

    # initial step
    tprev = ts
    tnext = ts + dt
    dprev = d[0, :]
    args = (A, B, dprev)
    state = solver.init(term, tprev, tnext, x0, args)

    # initialize output
    t_all = [tprev]
    x_all = jnp.array([x0])

    # main loop
    i = 0
    x = x0

    while tprev < te - dt:
        x, _, _, state, _ = solver.step(
            term, tprev, tnext, x, args, state, made_jump=False)
        tprev = tnext
        tnext = min(tprev + dt, te)

        # update disturbance for next step
        i += 1
        dnext = d[i, :]
        args = (A, B, dnext)

        # append results
        t_all.append(tnext)
        x_att = x.reshape(1, -1)
        x_all = jnp.concatenate([x_all, x_att], axis=0)

    return t_all, x_all

def get_disturbance_ph(disturbance, ts, PH, dt):
    """
    return disturbance for PH from a given data frame
    """
    return disturbance.loc[ts:ts+PH*dt, :]

if __name__ == '__main__':
    # MPC setting
    PH = 12
    CH = 1
    dt = 900

    # Get pre-stored disturbances generated from EPlus
    t_base = 181*24*3600 # 7/1
    dist = pd.read_csv('./data/disturbance_1min.csv', index_col=[0])
    n = len(dist)
    index = range(t_base, t_base + n*60, 60)
    dist.index = index
    dist = dist.groupby([dist.index // dt]).mean()
    index_dt = range(t_base, t_base + len(dist)*dt, dt)
    dist.index = index 
    print(dist.head())

    # Experiment settings
    ts = 195*24*3600
    te = 1*24*3600 + ts

    # get predictor
    predictor = {}
    # [Cai, Cwe, Cwi, Re, Ri, Rw, Rg]
    predictor['zone_model'] = jnp.array([6.9789902e+03, 2.1591113e+04, 1.8807944e+05, 3.4490612e+00,
                                      4.9556872e-01, 9.8289281e-02, 4.6257420e+00])
    # get zone model
    Cai, Cwe, Cwi, Re, Ri, Rw, Rg = predictor['zone_model']
    A, B, C, D = get_ABCD(Cai, Cwe, Cwi, Re, Ri, Rw, Rg)
    
    # initialize MPC
    mpc = MPC(PH, CH, dt, predictor)

    # initial state for rc zone
    state = jnp.array([20, 27.21, 26.76])
    
    # initialize output 
    u_opt  = []
    # main loop
    for t in range(ts, te, dt):
        
        # get disturbance
        dist_t_ph = get_disturbance_ph(dist, t, PH, dt)
        dist_t_ph = jnp.array(dist_t_ph.values)

        # set mpc states
        mpc.set_state(state)

        # mpc step
        u_ph = mpc.step(dist_t_ph)

        # get control to CH
        u_ch = u_ph[0]

        # apply control to system
        dist_t = dist_t_ph[0,:]
        dist_t = dist.at[0, 2].set(u_ch)
        args = (A, B, dist_t)
        ts, xs = forward(f, t, t+dt, dt, state, Euler(), args)
        state = xs[-1,:]
        
        # save results
        u_opt.append(u_ch)


    print(u_opt)
