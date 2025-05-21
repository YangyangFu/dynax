import jax
import jax.numpy as jnp
import unittest # Using unittest, but can also be pytest style if preferred by the project

from dynax.agents.mppi import MPPI
from dynax.core.base_block_state_space import BaseBlockSSM, BaseContinuousBlockSSM, BaseDiscreteBlockSSM

# --- Mock Components ---

class MockContinuousModel(BaseContinuousBlockSSM):
    state_dim: int
    input_dim: int
    output_dim: int # Though output (y) is not directly used by MPPI core, cost function might use it.

    # Make model configurable at initialization for testing different scenarios
    def __init__(self, state_dim=1, input_dim=1, output_dim=1, name=None):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        # No learnable parameters for this simple mock, so setup can be minimal
        # If the base class setup initializes _fxx etc. to None, that's fine.

    def setup(self):
        # If BaseContinuousBlockSSM expects certain attributes to be set up (like _fx, _fy), do it here.
        # For MPPI, only __call__ of the model is critical.
        pass

    def __call__(self, x, u):
        # Simple integrator: dx/dt = u
        # Assumes x and u are scalars or 1D vectors if state_dim/input_dim > 1
        if self.state_dim == 1 and self.input_dim == 1:
            dx = u
        elif self.input_dim == self.state_dim: # Vector case
            dx = u 
        else: # Default for mismatched dims, or specific test logic
            dx = jnp.zeros_like(x) + jnp.sum(u) # Example, can be refined
        
        y = x # Output is state
        return dx, y # Returns (d_state, output)

class MockDiscreteModel(BaseDiscreteBlockSSM):
    state_dim: int
    input_dim: int
    output_dim: int

    def __init__(self, state_dim=1, input_dim=1, output_dim=1, name=None):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

    def setup(self):
        pass

    def __call__(self, x, u):
        # Simple discrete model: x_next = x + u*dt (dt is part of MPPI agent)
        # Here, u is the direct change or scaled by a fixed internal dt if any.
        # Let's assume dt from MPPI is used in its Euler step for continuous,
        # so discrete model directly computes next state.
        if self.state_dim == 1 and self.input_dim == 1:
            x_next = x + u 
        elif self.input_dim == self.state_dim:
            x_next = x + u
        else:
            x_next = x + jnp.sum(u) # Example
            
        y = x # Output is current state before update
        return x_next, y # Returns (next_state, output)

def mock_cost_function(state, action, reference):
    # state: (state_dim,)
    # action: (input_dim,)
    # reference: (ref_dim,) or None -- ref_dim should match state_dim for this cost fn
    cost = 0.0
    # Penalize distance from reference if provided, else distance from zero
    if reference is not None:
        cost += jnp.sum((state - reference)**2)
    else:
        cost += jnp.sum(state**2) 
    # Penalize control effort
    cost += 0.1 * jnp.sum(action**2)
    return cost

# --- Test Class ---

class TestMPPIAgent(unittest.TestCase):

    def test_initialization_continuous_model(self):
        key = jax.random.PRNGKey(0)
        model = MockContinuousModel(state_dim=1, input_dim=1, output_dim=1)
        
        mppi_agent = MPPI(
            model=model,
            cost_function=mock_cost_function,
            horizon=10,
            num_samples=50, # Smaller for faster tests
            gamma=0.1,
            dt=0.1,
            input_dim=1,
            noise_covariance=jnp.array([[0.5]]),
            key_init=key
        )
        
        self.assertIsInstance(mppi_agent, MPPI)
        self.assertEqual(mppi_agent.horizon, 10)
        self.assertEqual(mppi_agent.num_samples, 50)
        self.assertTrue(jnp.array_equal(mppi_agent._key, key))
        self.assertEqual(mppi_agent._nominal_control_sequence.shape, (10, 1))
        self.assertTrue(jnp.all(mppi_agent._nominal_control_sequence == 0))

    def test_initialization_with_initial_nominal_sequence(self):
        key = jax.random.PRNGKey(1)
        model = MockDiscreteModel(state_dim=2, input_dim=1, output_dim=2)
        initial_seq = jnp.ones((5, 1)) * 0.5

        mppi_agent = MPPI(
            model=model,
            cost_function=mock_cost_function,
            horizon=5,
            num_samples=30,
            gamma=0.2,
            dt=0.2,
            input_dim=1,
            noise_covariance=jnp.array([[0.1]]),
            key_init=key,
            initial_nominal_sequence=initial_seq
        )
        self.assertTrue(jnp.array_equal(mppi_agent._nominal_control_sequence, initial_seq))

    def test_call_updates_state_and_returns_action_continuous(self):
        key = jax.random.PRNGKey(42)
        model = MockContinuousModel(state_dim=1, input_dim=1, output_dim=1)
        
        mppi_agent = MPPI(
            model=model,
            cost_function=mock_cost_function,
            horizon=5,
            num_samples=20, # Low samples for speed
            gamma=1.0,
            dt=0.1,
            input_dim=1,
            noise_covariance=jnp.array([[0.2]]),
            key_init=key
        )
        
        initial_key_val = mppi_agent._key
        initial_nominal_seq_val = mppi_agent._nominal_control_sequence.copy()
        
        current_state = jnp.array([1.0]) # Single state dimension
        action = mppi_agent(current_state) # Call the agent
        
        self.assertEqual(action.shape, (1,)) # input_dim
        self.assertFalse(jnp.array_equal(mppi_agent._key, initial_key_val))
        # Nominal sequence should also change after the call
        self.assertFalse(jnp.allclose(mppi_agent._nominal_control_sequence, initial_nominal_seq_val)) 

    def test_call_updates_state_and_returns_action_discrete(self):
        key = jax.random.PRNGKey(100)
        model = MockDiscreteModel(state_dim=2, input_dim=2, output_dim=2)
        
        mppi_agent = MPPI(
            model=model,
            cost_function=mock_cost_function,
            horizon=3,
            num_samples=10,
            gamma=0.5,
            dt=1.0, # dt is used by continuous model integration, less direct for discrete
            input_dim=2,
            noise_covariance=jnp.array([[0.5, 0], [0, 0.5]]),
            key_init=key
        )
        
        initial_key_val = mppi_agent._key
        initial_nominal_seq_val = mppi_agent._nominal_control_sequence.copy()

        current_state = jnp.array([0.5, -0.5])
        reference = jnp.array([0.0, 0.0]) # Target zero state
        action = mppi_agent(current_state, reference_schedule=reference)
        
        self.assertEqual(action.shape, (2,))
        self.assertFalse(jnp.array_equal(mppi_agent._key, initial_key_val))
        self.assertFalse(jnp.allclose(mppi_agent._nominal_control_sequence, initial_nominal_seq_val))

    # TODO: Add a more functional test if possible, e.g., action has correct sign
    # This can be complex to make robust for a unit test.
    # For example: if current_state is positive, and cost wants it to be zero,
    # and model is dx = u, then action 'u' should be negative.

    def test_action_sign_simple_case(self):
        # Test if action pushes towards zero for a simple 1D case
        key = jax.random.PRNGKey(7)
        model = MockContinuousModel(state_dim=1, input_dim=1, output_dim=1) # dx/dt = u
        
        # Cost function: state^2 + 0.01*action^2 (penalize non-zero state, small penalty on action)
        def simple_cost(state, action, reference):
            return jnp.sum(state**2) + 0.01 * jnp.sum(action**2)

        mppi_agent = MPPI(
            model=model,
            cost_function=simple_cost,
            horizon=5,
            num_samples=100, # More samples for better convergence to expected behavior
            gamma=0.05,      # Smaller gamma -> more exploitation of good samples
            dt=0.1,
            input_dim=1,
            noise_covariance=jnp.array([[0.1]]), # Small noise
            key_init=key
        )

        current_state_positive = jnp.array([2.0])
        action_positive_state = mppi_agent(current_state_positive)
        # For dx/dt = u, to reduce positive state, u should be negative
        self.assertTrue(action_positive_state[0] < 0, 
                        f"Action {action_positive_state[0]} should be negative for positive state {current_state_positive[0]}")

        # Reset key and nominal sequence for a comparable next call (if __init__ was re-run)
        # Since the agent is stateful, subsequent calls build on previous state.
        # For isolated test cases on __call__, re-initialize or manage state carefully.
        # Here, we are testing sequential calls and their effects on the *same* agent instance is fine.
        # To test from a "clean slate" for the negative state, we would re-init MPPI.
        # Let's re-initialize for a clean test of the negative state scenario.
        mppi_agent_neg = MPPI(
            model=model, cost_function=simple_cost, horizon=5, num_samples=100,
            gamma=0.05, dt=0.1, input_dim=1, noise_covariance=jnp.array([[0.1]]),
            key_init=jax.random.PRNGKey(8) # Different key for good measure
        )
        current_state_negative = jnp.array([-2.0])
        action_negative_state = mppi_agent_neg(current_state_negative)
        # For dx/dt = u, to increase negative state (towards zero), u should be positive
        self.assertTrue(action_negative_state[0] > 0,
                        f"Action {action_negative_state[0]} should be positive for negative state {current_state_negative[0]}")


if __name__ == '__main__':
    unittest.main()
```
