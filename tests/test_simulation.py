"""
Unit tests for simulation module.
"""

import pytest
import numpy as np
from src.mdp_inventory import InventoryMDP, InventoryParams
from src.simulation import simulate_policy
from src.baseline_policies import make_sS_policy


@pytest.fixture
def test_mdp():
    """Create test MDP."""
    return InventoryMDP(InventoryParams(
        max_inventory=5,
        max_order=3,
        demand_probs={0: 0.3, 1: 0.7},
        holding_cost=0.1,
        shortage_cost=1.0,
        order_cost=0.2,
        discount=0.95
    ))


class TestSimulatePolicy:
    """Test simulate_policy function."""
    
    def test_basic_simulation(self, test_mdp):
        """Test basic simulation runs."""
        policy = make_sS_policy(s=2, S=4)
        metrics = simulate_policy(test_mdp, policy, T=10, seed=42)
        
        assert 'total_cost' in metrics
        assert 'avg_cost_per_period' in metrics
        assert 'service_level' in metrics
        assert metrics['T'] == 10
    
    def test_reproducibility(self, test_mdp):
        """Test that simulation is reproducible with seed."""
        policy = make_sS_policy(s=2, S=4)
        metrics1 = simulate_policy(test_mdp, policy, T=50, seed=123)
        metrics2 = simulate_policy(test_mdp, policy, T=50, seed=123)
        
        assert metrics1['total_cost'] == metrics2['total_cost']
        assert np.array_equal(metrics1['demand_sequence'], metrics2['demand_sequence'])
    
    def test_invalid_initial_state(self, test_mdp):
        """Test that invalid initial state raises error."""
        policy = make_sS_policy(s=2, S=4)
        with pytest.raises(ValueError):
            simulate_policy(test_mdp, policy, T=10, initial_state=10)  # > max_inventory
    
    def test_invalid_T(self, test_mdp):
        """Test that invalid T raises error."""
        policy = make_sS_policy(s=2, S=4)
        with pytest.raises(ValueError):
            simulate_policy(test_mdp, policy, T=0)  # Invalid
    
    def test_metrics_structure(self, test_mdp):
        """Test that all expected metrics are present."""
        policy = make_sS_policy(s=2, S=4)
        metrics = simulate_policy(test_mdp, policy, T=20, seed=42)
        
        required_keys = [
            'total_reward', 'total_cost', 'avg_cost_per_period',
            'total_demand', 'fulfilled_demand', 'unmet_demand',
            'service_level', 'per_step_costs', 'per_step_holding',
            'per_step_shortage', 'per_step_ordering', 'rewards',
            'inventory_levels', 'demand_sequence', 'actions', 'T'
        ]
        
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
    
    def test_service_level_calculation(self, test_mdp):
        """Test service level calculation."""
        policy = make_sS_policy(s=2, S=4)
        metrics = simulate_policy(test_mdp, policy, T=100, seed=42)
        
        # Service level should be between 0 and 1
        assert 0 <= metrics['service_level'] <= 1
        
        # Should equal fulfilled / total
        if metrics['total_demand'] > 0:
            expected = metrics['fulfilled_demand'] / metrics['total_demand']
            assert abs(metrics['service_level'] - expected) < 1e-10


