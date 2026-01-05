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


class TestSolverSanity:
    """Sanity tests on tiny MDPs with known optimal solutions."""
    
    def test_tiny_deterministic_mdp(self):
        """
        Test on a tiny 2-state MDP with deterministic demand.
        
        MDP setup:
        - States: 0, 1 (max_inventory=1)
        - Actions: 0, 1 (max_order=1)
        - Demand: always 1 (deterministic)
        - Costs: holding=0.1, shortage=1.0, order=0.2
        - Discount: 0.9
        
        Hand calculation:
        State 0:
          - Action 0: stay at 0, cost = shortage(1.0) = 1.0, reward = -1.0
          - Action 1: go to 1 then to 0, cost = order(0.2) + holding(0.1) = 0.3, reward = -0.3
          - Optimal: action 1 (order)
        
        State 1:
          - Action 0: go to 0, cost = 0, reward = 0
          - Action 1: stay at 1, cost = order(0.2) + holding(0.1) = 0.3, reward = -0.3
          - Optimal: action 0 (don't order)
        
        Expected optimal policy: [1, 0] (order in state 0, don't order in state 1)
        """
        mdp = InventoryMDP(InventoryParams(
            max_inventory=1,
            max_order=1,
            demand_probs={1: 1.0},  # Deterministic demand of 1
            holding_cost=0.1,
            shortage_cost=1.0,
            order_cost=0.2,
            discount=0.9
        ))
        
        # Test Value Iteration
        V_vi, policy_vi, _ = value_iteration(mdp, tol=1e-6)
        assert policy_vi[0] == 1, f"State 0: expected action 1, got {policy_vi[0]}"
        assert policy_vi[1] == 0, f"State 1: expected action 0, got {policy_vi[1]}"
        
        # Test Policy Iteration
        V_pi, policy_pi, _ = policy_iteration(mdp)
        assert policy_pi[0] == 1, f"State 0: expected action 1, got {policy_pi[0]}"
        assert policy_pi[1] == 0, f"State 1: expected action 0, got {policy_pi[1]}"
        
        # Both methods should agree
        assert np.array_equal(policy_vi, policy_pi), \
            f"VI and PI policies differ: VI={policy_vi}, PI={policy_pi}"
        
        # Value functions should be close
        assert np.max(np.abs(V_vi - V_pi)) < 1e-4, \
            f"VI and PI value functions differ: max_diff={np.max(np.abs(V_vi - V_pi))}"
        
        # Verify Bellman equation holds (sanity check)
        # For state 0, action 1 should be optimal
        Q_0_0 = mdp.R_sa[0, 0] + mdp.discount * np.sum(mdp.P[0, 0, :] * V_vi)
        Q_0_1 = mdp.R_sa[0, 1] + mdp.discount * np.sum(mdp.P[0, 1, :] * V_vi)
        assert Q_0_1 >= Q_0_0 - 1e-6, \
            f"State 0: Q(0,1)={Q_0_1} should be >= Q(0,0)={Q_0_0}"
    
    def test_tiny_stochastic_mdp(self):
        """
        Test on a tiny 2-state MDP with stochastic demand.
        
        MDP setup:
        - States: 0, 1 (max_inventory=1)
        - Actions: 0, 1 (max_order=1)
        - Demand: {0: 0.5, 1: 0.5}
        - Costs: holding=0.1, shortage=1.0, order=0.2
        - Discount: 0.9
        
        This test verifies that the reward/probability calculation is correct
        by checking that both VI and PI converge to the same policy.
        """
        mdp = InventoryMDP(InventoryParams(
            max_inventory=1,
            max_order=1,
            demand_probs={0: 0.5, 1: 0.5},
            holding_cost=0.1,
            shortage_cost=1.0,
            order_cost=0.2,
            discount=0.9
        ))
        
        # Both methods should converge
        V_vi, policy_vi, iters_vi = value_iteration(mdp, tol=1e-6)
        V_pi, policy_pi, iters_pi = policy_iteration(mdp)
        
        assert iters_vi < 1000, "Value iteration should converge quickly"
        assert iters_pi < 100, "Policy iteration should converge quickly"
        
        # Both methods should agree
        assert np.array_equal(policy_vi, policy_pi), \
            f"VI and PI policies differ: VI={policy_vi}, PI={policy_pi}"
        
        # Value functions should be very close
        assert np.max(np.abs(V_vi - V_pi)) < 1e-4, \
            f"VI and PI value functions differ: max_diff={np.max(np.abs(V_vi - V_pi))}"
        
        # Verify Bellman residual is small (sanity check)
        for s in range(mdp.num_states()):
            a = policy_vi[s]
            Q_sa = mdp.R_sa[s, a] + mdp.discount * np.sum(mdp.P[s, a, :] * V_vi)
            bellman_residual = abs(V_vi[s] - Q_sa)
            assert bellman_residual < 1e-5, \
                f"State {s}: Bellman residual {bellman_residual} too large"

