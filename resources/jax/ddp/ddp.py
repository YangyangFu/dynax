import jax.numpy as jnp
import numpy as np 
from jax import grad, jit, vmap

class DifferentiableDynamicProgrammingPolicy():
    """ 
    Set up and solve a trajectory optimization problem as described below using differentiable dynamic programming.

    min_{u} \sum_{t=0}^{T-1} L(x_t, u_t) + Lf(x_T)
    s.t. x_{t+1} = f(x_t, u_t), x_0 = x_0

    where L is the stage cost, Lf is the terminal cost, f is the dynamics, and x_0 is the initial state.
    """
    def __init__(self, discrete_system, num_timesteps):
        self.system = discrete_system 
        self.num_timesteps = num_timesteps
    
    def setInitialStates(self, initial_states):
        self.initial_states = initial_states
    
    def setTargetStates(self, target_states):
        self.target_states = target_states

    def setStageCost(self):
        pass

    def setTerminalCost(self):
        pass

    def setControlLimits(self):
        pass

    def _calc_dynamics(self, x, u):
        """
        Calculate the dynamics of the system at a given state and control input.

        xnext = f(x, u)
        """
        return self.system.dynamics(x, u)
    
    def _calc_stage_cost(self, x, u):
        """
        Calculate the stage cost of the system at a given state and control input.

        L = L(x, u)
        """
        return self.system.stage_cost(x, u)
    
    def _calc_terminal_cost(self, x):
        """
        Calculate the terminal cost of the system at a given state.

        Lf = Lf(x)
        """
        return self.system.terminal_cost(x)
    
    def _calc_dynamics_jacobian(self, x, u):
        """
        Calculate the jacobian of the dynamics of the system at a given state and control input.

        df/dx = dfdx(x, u)
        """
        return self.system.dynamics_jacobian(x, u)

    def _calc_stage_cost_jacobian(self, x, u):
        """
        Calculate the jacobian of the stage cost of the system at a given state and control input.

        dL/dx = dLdx(x, u)
        dL/du = dLdu(x, u)
        """
        return self.system.stage_cost_jacobian(x, u)
    
    def _calc_terminal_cost_jacobian(self, x):
        """
        Calculate the jacobian of the terminal cost of the system at a given state.

        dLf/dx = dLfdx(x)
        """
        return self.system.terminal_cost_jacobian(x)
    
    def _calc_dynamics_hessian(self, x, u):
        """
        Calculate the hessian of the dynamics of the system at a given state and control input.

        d2f/dx2 = d2fdx2(x, u)
        """
        return self.system.dynamics_hessian(x, u)

    def _calc_stage_cost_hessian(self, x, u):
        """
        Calculate the hessian of the stage cost of the system at a given state and control input.

        d2L/dx2 = d2Ldx2(x, u)
        d2L/du2 = d2Ldu2(x, u)
        d2L/dudx = d2Ldudx(x, u)
        """
        return self.system.stage_cost_hessian(x, u)
    
    def _calc_terminal_cost_hessian(self, x):
        """
        Calculate the hessian of the terminal cost of the system at a given state.

        d2Lf/dx2 = d2Lfdx2(x)
        """
        return self.system.terminal_cost_hessian(x)
    
    # this is from https://github.com/vincekurtz/drake_ddp/blob/73a0cd8a78bbf788dba3e11f840f2f90ec6baf6d/ilqr.py#L256.
    def _linesearch(self, L_last):
            """
            Determine a value of eps in (0,1] that results in a suitably
            reduced cost, based on forward simulations of the system.
            This involves simulating the system according to the control law
                u = u_bar - eps*kappa - K*(x-x_bar).
            and reducing eps by a factor of beta until the improvement in
            total cost is greater than gamma*(expected cost reduction)
            Args:
                L_last: Total cost from the last iteration.
            Returns:
                eps:        Linesearch parameter
                x:          (n,N) numpy array of new states
                u:          (m,N-1) numpy array of new control inputs
                L:          Total cost/loss associated with the new trajectory
                n_iters:    Number of linesearch iterations taken
            Raises:
                RuntimeError: if eps has been reduced to <1e-8 and we still
                            haven't found a suitable parameter.
            """
            eps = 1.0
            n_iters = 0
            while eps >= 1e-8:
                n_iters += 1

                # Simulate system forward using the given eps value
                L = 0
                expected_improvement = 0
                x = np.zeros((self.n,self.N))
                u = np.zeros((self.m,self.N-1))

                x[:,0] = self.x0
                for t in range(0,self.N-1):
                    u[:,t] = self.u_bar[:,t] - eps*self.kappa[:,t] - self.K[:,:,t]@(x[:,t] - self.x_bar[:,t])
                    
                    try:
                        x[:,t+1] = self._calc_dynamics(x[:,t], u[:,t])
                    except RuntimeError as e:
                        # If dynamics are infeasible, consider the loss to be infinite 
                        # and stop simulating. This will lead to a reduction in eps
                        print("Warning: encountered infeasible simulation in linesearch")
                        #print(e)
                        L = np.inf
                        break

                    L += (x[:,t]-self.x_nom).T@self.Q@(x[:,t]-self.x_nom) + u[:,t].T@self.R@u[:,t]
                    expected_improvement += -eps*(1-eps/2)*self.dV_coeff[t]
                L += (x[:,-1]-self.x_nom).T@self.Qf@(x[:,-1]-self.x_nom)

                # Chech whether the improvement is sufficient
                improvement = L_last - L
                if improvement > self.gamma*expected_improvement:
                    return eps, x, u, L, n_iters

                # Otherwise reduce eps by a factor of beta
                eps *= self.beta

            raise RuntimeError("linesearch failed after %s iterations"%n_iters)

    def backward_pass(self):
        pass

    def forward_pass(self):
        pass


    def solve(self):
        """
        Solve the optimization problem and return the (locally) optimal
        state and input trajectories. 
        Return:
            x:              (n,N) numpy array containing optimal state trajectory
            u:              (m,N-1) numpy array containing optimal control tape
            solve_time:     Total solve time in seconds
            optimal_cost:   Total cost associated with the (locally) optimal solution
        """

        pass 
    
        
