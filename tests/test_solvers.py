"""
Unit tests for solvers module.
"""

import pytest
import numpy as np
from src.mdp_inventory import InventoryMDP, InventoryParams
from src.solvers import value_iteration, policy_iteration, policy_evaluation


@pytest.fixture
def simple_mdp():
    """Create a simple MDP for testing."""
    return InventoryMDP(InventoryParams(
        max_inventory=3,
        max_order=2,
        demand_probs={0: 0.5, 1: 0.5},
        holding_cost=0.1,
        shortage_cost=1.0,
        order_cost=0.2,
        discount=0.9
    ))


class TestValueIteration:
    """Test value iteration algorithm."""
    
    def test_convergence(self, simple_mdp):
        """Test that value iteration converges."""
        V, policy, iters = value_iteration(simple_mdp, tol=1e-6)
        assert iters > 0
        assert iters < 10000  # Should converge quickly
        assert V.shape == (simple_mdp.num_states(),)
        assert policy.shape == (simple_mdp.num_states(),)
    
    def test_policy_validity(self, simple_mdp):
        """Test that returned policy is valid."""
        V, policy, _ = value_iteration(simple_mdp)
        # Check policy actions are in valid range
        assert np.all(policy >= 0)
        assert np.all(policy < simple_mdp.num_actions())
    
    def test_value_function_monotonicity(self, simple_mdp):
        """Test that value function is reasonable."""
        V, _, _ = value_iteration(simple_mdp)
        # Value should be finite
        assert np.all(np.isfinite(V))
        # For inventory problems, higher inventory often has better value
        # (though not always due to holding costs)


class TestPolicyIteration:
    """Test policy iteration algorithm."""
    
    def test_convergence(self, simple_mdp):
        """Test that policy iteration converges."""
        V, policy, iters = policy_iteration(simple_mdp)
        assert iters > 0
        assert iters < 1000  # Should converge quickly
        assert V.shape == (simple_mdp.num_states(),)
        assert policy.shape == (simple_mdp.num_states(),)
    
    def test_policy_validity(self, simple_mdp):
        """Test that returned policy is valid."""
        V, policy, _ = policy_iteration(simple_mdp)
        assert np.all(policy >= 0)
        assert np.all(policy < simple_mdp.num_actions())
    
    def test_consistency_with_value_iteration(self, simple_mdp):
        """Test that policy iteration and value iteration give similar results."""
        V_vi, policy_vi, _ = value_iteration(simple_mdp, tol=1e-6)
        V_pi, policy_pi, _ = policy_iteration(simple_mdp)
        
        # Policies should be identical (or very similar)
        assert np.array_equal(policy_vi, policy_pi)
        
        # Value functions should be very close
        assert np.max(np.abs(V_vi - V_pi)) < 1e-4


class TestPolicyEvaluation:
    """Test policy evaluation function."""
    
    def test_evaluation(self, simple_mdp):
        """Test policy evaluation."""
        # Use a simple policy: order 1 if inventory < 2, else 0
        policy = np.array([1, 1, 0, 0])
        V = policy_evaluation(simple_mdp, policy)
        assert V.shape == (simple_mdp.num_states(),)
        assert np.all(np.isfinite(V))
    
    def test_invalid_policy_shape(self, simple_mdp):
        """Test that invalid policy shape raises error."""
        policy = np.array([0, 1])  # Wrong shape
        with pytest.raises(ValueError):
            policy_evaluation(simple_mdp, policy)
    
    def test_invalid_policy_actions(self, simple_mdp):
        """Test that invalid actions raise error."""
        policy = np.array([0, 1, 2, 10])  # Action 10 is invalid
        with pytest.raises(ValueError):
            policy_evaluation(simple_mdp, policy)

