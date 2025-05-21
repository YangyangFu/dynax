import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional

from dynax.agents.base import BaseAgent
from dynax.core.base_block_state_space import BaseBlockSSM, BaseContinuousBlockSSM, BaseDiscreteBlockSSM

class MPPI(BaseAgent):
    # --- Hyperparameters (Fixed after initialization) ---
    # These parameters are set once during instantiation and are not changed by the agent's __call__ method.
    model: BaseBlockSSM
    cost_function: Callable[[jax.Array, jax.Array, Optional[jax.Array]], jax.Array]
    horizon: int
    num_samples: int
    gamma: float # Temperature parameter
    dt: float
    input_dim: int
    noise_covariance: jax.Array # Shape: (input_dim, input_dim)
    
    # --- Internal State ---
    # These attributes make instances of MPPI stateful. They are modified during the __call__ method.
    # _key: JAX PRNGKey. It's updated in __call__, allowing for reproducible randomness if the initial key is fixed.
    # _nominal_control_sequence: The sequence around which new samples are generated. Updated in __call__.
    # 
    # Note on JAX JIT Compilation and State:
    # Direct mutation of these attributes within __call__ (e.g., self._key = new_key) means that
    # if `mppi_agent.__call__` is JIT-compiled, JAX's default behavior (treating `self` as static for
    # instance methods unless part of a Flax module's state variables) might lead to the state updates
    # not persisting across calls as one might expect in pure Python.
    # For robust JIT-compatibility with state in Flax, one would typically define these stateful attributes
    # in the `setup` method using `self.variable('state', ...)` and then access/update them using
    # `self.variables['state']...` in conjunction with `mutable=['state']` (or `mutable=True`) in the `apply`
    # method (or a method called by `apply`).
    # However, for this MPPI agent, which inherits from BaseAgent (an nn.Module), direct state mutation 
    # is used for simplicity. This approach is common for higher-level agent objects where the __call__ 
    # method itself might not be the primary JIT boundary, but rather internal, functionally pure 
    # components (like the rollout and cost calculation, which are vmapped and can be JITted).
    _key: jax.random.PRNGKey 
    _nominal_control_sequence: jax.Array

    def __init__(self, model: BaseBlockSSM, cost_function: Callable, horizon: int,
                 num_samples: int, gamma: float, dt: float, input_dim: int,
                 noise_covariance: jax.Array, key_init: jax.random.PRNGKey,
                 initial_nominal_sequence: Optional[jax.Array] = None):
        """
        Initializes the MPPI agent.

        The parameters passed here are treated as fixed hyperparameters for the agent instance.
        Internal states like the PRNG key and the nominal control sequence are initialized
        and then updated during subsequent calls to the agent.

        Args:
            model: The system model (instance of BaseBlockSSM, e.g., BaseContinuousBlockSSM or BaseDiscreteBlockSSM).
            cost_function: A callable that computes the cost. Signature: (state, input, reference_at_t?) -> scalar_cost.
            horizon: The prediction horizon (number of timesteps, H).
            num_samples: The number of control sequences to sample in each optimization step (K).
            gamma: The temperature parameter for the MPPI algorithm (lambda in some notations).
            dt: The time step duration for the discrete-time simulation.
            input_dim: The dimensionality of the control input vector.
            noise_covariance: The covariance matrix (input_dim x input_dim) for the Gaussian noise
                              used to generate control sequence samples.
            key_init: The initial JAX PRNGKey for all random operations.
            initial_nominal_sequence: An optional initial nominal control sequence of shape (horizon, input_dim).
                                      If None, it defaults to a sequence of zeros.
        """
        super().__init__(name=None) # MPPI is an nn.Module via BaseAgent

        # Store hyperparameters
        self.model = model
        self.cost_function = cost_function
        self.horizon = horizon
        self.num_samples = num_samples
        self.gamma = gamma
        self.dt = dt
        self.input_dim = input_dim
        self.noise_covariance = noise_covariance
        
        # Initialize internal state attributes
        self._key = key_init # This key will be updated in __call__
        if initial_nominal_sequence is None:
            self._nominal_control_sequence = jnp.zeros((self.horizon, self.input_dim))
        else:
            expected_shape = (self.horizon, self.input_dim)
            if initial_nominal_sequence.shape != expected_shape:
                raise ValueError(
                    f"Initial nominal sequence has incorrect shape. Expected {expected_shape}, "
                    f"got {initial_nominal_sequence.shape}"
                )
            self._nominal_control_sequence = initial_nominal_sequence

    def __call__(self, current_state: jax.Array, reference_schedule: Optional[jax.Array] = None) -> jax.Array:
        """
        Computes the optimal control action using Model Predictive Path Integral control.
        
        This method is stateful: it mutates the agent's internal PRNG key (`self._key`) and
        the nominal control sequence (`self._nominal_control_sequence`) at each call.
        Please refer to the "Note on JAX JIT Compilation and State" in the class comments
        for implications regarding JAX's JIT compilation.
        """
        # Update internal PRNG key: split the current key to get a new key for this call's operations,
        # and update self._key to be used in the next call. This makes the randomness sequence
        # dependent on the history of calls.
        call_key, self._key = jax.random.split(self._key)

        # The nominal sequence for sampling is the one stored and updated from the previous optimization step.
        nominal_seq_for_sampling = self._nominal_control_sequence

        # --- 1. Sample Control Sequences ---
        mean_noise = jnp.zeros(self.input_dim)
        # Generate noise: (num_samples, horizon, input_dim)
        noise = jax.random.multivariate_normal(
            call_key, # Use the key specifically split for this __call__ invocation
            mean=mean_noise,
            cov=self.noise_covariance,
            shape=(self.num_samples, self.horizon) 
        )
        # Add noise to the nominal sequence to get sampled sequences
        # sampled_control_sequences shape: (num_samples, horizon, input_dim)
        sampled_control_sequences = nominal_seq_for_sampling + noise 

        # --- 2. Simulate Trajectories ---
        # This inner function performs a rollout for a single control sequence.
        def perform_rollout(control_sequence_sample, initial_state_rollout):
            # Inner scan function for stepping through the model dynamics
            def dynamics_step(state_at_step, action_at_step):
                # Determine if model is continuous or discrete for state update
                if isinstance(self.model, BaseContinuousBlockSSM):
                    d_state, _ = self.model(state_at_step, action_at_step) # model returns (dx, y)
                    next_state = state_at_step + self.dt * d_state
                elif isinstance(self.model, BaseDiscreteBlockSSM):
                    next_state, _ = self.model(state_at_step, action_at_step) # model returns (x_next, y)
                else:
                    # This error helps users if they pass a model not deriving from the expected bases.
                    raise NotImplementedError(
                        "Model type must be an instance of BaseContinuousBlockSSM or BaseDiscreteBlockSSM. "
                        "Ensure your model inherits from one of these."
                    )
                return next_state, next_state # (carry_for_scan, output_of_scan_step)
            
            # Use jax.lax.scan for efficient rollout over the horizon
            _, rollout_states = jax.lax.scan(dynamics_step, initial_state_rollout, control_sequence_sample)
            # rollout_states shape: (horizon, state_dim)
            return rollout_states

        # Vectorize the rollout over all sampled control sequences using jax.vmap.
        # all_rollout_states shape: (num_samples, horizon, state_dim)
        all_rollout_states = jax.vmap(perform_rollout, in_axes=(0, None))(
            sampled_control_sequences, current_state
        )

        # --- 3. Calculate Cost for Each Trajectory ---
        # This inner function calculates the total cost for a single trajectory.
        def calculate_total_trajectory_cost(single_rollout_states, single_control_sequence, ref_schedule_for_sample):
            # single_rollout_states: (horizon, state_dim)
            # single_control_sequence: (horizon, input_dim)
            # ref_schedule_for_sample: (horizon, ref_dim) or None
            
            # Vmap the cost_function over the horizon for a single trajectory
            if ref_schedule_for_sample is not None:
                costs_per_step = jax.vmap(self.cost_function, in_axes=(0, 0, 0))(
                    single_rollout_states, single_control_sequence, ref_schedule_for_sample
                )
            else:
                # If no reference schedule, pass None to each call of cost_function.
                costs_per_step = jax.vmap(self.cost_function, in_axes=(0, 0, None))(
                    single_rollout_states, single_control_sequence, None 
                )
            return jnp.sum(costs_per_step)
        
        # Process reference_schedule to ensure it's compatible with the horizon for cost calculation.
        processed_ref_schedule = None
        if reference_schedule is not None:
            if reference_schedule.ndim == 1: # Single reference point, tile across horizon
                processed_ref_schedule = jnp.tile(reference_schedule[jnp.newaxis, :], (self.horizon, 1))
            elif reference_schedule.ndim == 2:
                if reference_schedule.shape[0] == self.horizon: # Already shaped for horizon
                    processed_ref_schedule = reference_schedule
                elif reference_schedule.shape[0] == 1: # Single ref point provided as 2D array, tile
                    processed_ref_schedule = jnp.tile(reference_schedule, (self.horizon, 1))
                else: # Incorrect number of steps for a per-horizon schedule
                    raise ValueError(
                        f"Reference schedule has {reference_schedule.shape[0]} steps, but horizon is {self.horizon}."
                    )
            else: # Incorrect dimensions
                raise ValueError(
                    f"Reference schedule has an incompatible shape: {reference_schedule.shape}. "
                    "Expected (ref_dim,) or (horizon, ref_dim) or (1, ref_dim)."
                )

        # Vectorize cost calculation over all samples.
        # all_total_costs shape: (num_samples,)
        all_total_costs = jax.vmap(calculate_total_trajectory_cost, in_axes=(0, 0, None))(
            all_rollout_states, sampled_control_sequences, processed_ref_schedule
        )
        
        # --- 4. Compute Importance Sampling Weights ---
        min_cost = jnp.min(all_total_costs) # Subtract min_cost for numerical stability before exponentiation.
        # exp_term shape: (num_samples,)
        exp_term = jnp.exp(-1.0 / self.gamma * (all_total_costs - min_cost))
        sum_exp_term = jnp.sum(exp_term)
        
        # Avoid division by zero if all exp_terms are zero (e.g., gamma is tiny or costs are huge -> exp underflows).
        # If sum_exp_term is zero, fall back to uniform weights. This also handles num_samples = 1.
        is_sum_zero = (sum_exp_term == 0.0)
        # Use a safe denominator: if sum_exp_term is 0, use 1.0 to prevent NaN.
        safe_sum_exp_term = jnp.where(is_sum_zero, 1.0, sum_exp_term)
        normalized_weights = exp_term / safe_sum_exp_term
        # If the original sum was zero, normalized_weights are now all zero. Correct by setting to uniform.
        normalized_weights = jnp.where(
            is_sum_zero, 
            jnp.ones_like(exp_term) / self.num_samples, 
            normalized_weights
        )

        # --- 5. Compute Weighted Average of Sampled Control Sequences ---
        # optimal_control_sequence shape: (horizon, input_dim)
        optimal_control_sequence = jnp.sum(
            # Expand dims of weights for broadcasting: (num_samples, 1, 1)
            normalized_weights[:, jnp.newaxis, jnp.newaxis] * sampled_control_sequences, 
            axis=0 # Sum over the samples dimension
        )

        # --- 6. Update Nominal Control Sequence for Next Step (Stateful Update) ---
        # The current nominal sequence (`self._nominal_control_sequence`) is updated for the *next* call.
        # This is done by shifting the calculated optimal sequence: the first action is applied,
        # and the remainder (shifted) forms the start of the new nominal sequence.
        # The last action of this new nominal sequence is typically a repeat of the previous last optimal action.
        self._nominal_control_sequence = jnp.roll(optimal_control_sequence, shift=-1, axis=0)
        self._nominal_control_sequence = self._nominal_control_sequence.at[-1, :].set(optimal_control_sequence[-1, :]) # Repeat last element
        
        # Return the first action of the optimal sequence
        return optimal_control_sequence[0]

if __name__ == '__main__':
    # This is a conceptual example of how to use the MPPI agent.
    # It's not intended to be run directly as part of the library,
    # but illustrates the setup and usage pattern.

    import jax
    import jax.numpy as jnp
    from dynax.core.base_block_state_space import BaseContinuousBlockSSM # Or a specific model

    # 1. Define a system model
    # Using a very simple 1D continuous model for demonstration: dx/dt = u
    class SimpleContinuousModel(BaseContinuousBlockSSM):
        state_dim: int = 1
        input_dim: int = 1
        output_dim: int = 1 # Output y = x

        def __init__(self, name=None):
            super().__init__(name=name)
            # No learnable parameters needed for this simple model's setup.
            # Base class handles setup of _fxx etc. if they are not overridden.

        def __call__(self, x, u):
            # x: state (1D), u: input (1D)
            dx = u  # Simple integrator dynamics
            y = x   # Output is the state
            return dx, y

    model_instance = SimpleContinuousModel()

    # 2. Define a cost function
    # Cost penalizes deviation from a zero state and control effort.
    def example_cost_fn(state, action, reference_state):
        # reference_state is optional; here, we assume it's the target state (e.g., jnp.array([0.0]))
        # or None if no specific reference is tracked beyond origin.
        cost = jnp.sum(state**2) # Penalize non-zero state
        cost += 0.1 * jnp.sum(action**2) # Penalize control effort
        if reference_state is not None: # Example of using a reference
             cost += jnp.sum((state - reference_state)**2) # Additional penalty for deviation from ref
        return cost

    # 3. Initialize the MPPI agent
    key = jax.random.PRNGKey(0)
    mppi_params = {
        'model': model_instance,
        'cost_function': example_cost_fn,
        'horizon': 15,
        'num_samples': 200,
        'gamma': 0.1,
        'dt': 0.1, # Timestep for MPPI's internal simulation and action application
        'input_dim': 1,
        'noise_covariance': jnp.array([[0.5]]), # Covariance for sampling noise
        'key_init': key,
        'initial_nominal_sequence': jnp.zeros((15, 1)) # (horizon, input_dim)
    }
    mppi_agent = MPPI(**mppi_params)

    # 4. Conceptual simulation loop
    print("Conceptual MPPI Usage Example:")
    current_state = jnp.array([2.0]) # Initial state of the system
    print(f"Initial state: {current_state}")

    num_simulation_steps = 50
    for i in range(num_simulation_steps):
        # Get action from MPPI agent
        # A reference schedule could be passed if the cost function uses it.
        # For this example, let's assume a fixed reference (target state) of 0.0.
        # If the cost_function expects a reference of shape (horizon, ref_dim),
        # it should be prepared accordingly. Here, example_cost_fn handles scalar reference.
        target_state_for_horizon = jnp.array([0.0]) # Target state is 0.0

        action = mppi_agent(current_state, reference_schedule=target_state_for_horizon)
        
        # Conceptually apply action to the real system (or a separate simulation model)
        # For this example, we'll use the model's dynamics directly for simplicity.
        # This is what MPPI does internally, but here it's the "environment" step.
        if isinstance(mppi_agent.model, BaseContinuousBlockSSM):
            d_state, _ = mppi_agent.model(current_state, action)
            current_state = current_state + mppi_agent.dt * d_state
        elif isinstance(mppi_agent.model, BaseDiscreteBlockSSM):
            current_state, _ = mppi_agent.model(current_state, action)
        
        print(f"Step {i+1}: Action = {action}, New State = {current_state}")

        if jnp.allclose(current_state, target_state_for_horizon, atol=0.1):
            print(f"Target state reached at step {i+1}.")
            break
    print("Simulation finished.")
```
