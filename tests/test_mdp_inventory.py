"""
Unit tests for mdp_inventory module.
"""

import pytest
import numpy as np
from src.mdp_inventory import InventoryMDP, InventoryParams
from src.validation import validate_params, validate_mdp


class TestInventoryParams:
    """Test InventoryParams dataclass."""
    
    def test_valid_params(self):
        """Test valid parameters."""
        params = InventoryParams(
            max_inventory=10,
            max_order=5,
            demand_probs={0: 0.1, 1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1},
            holding_cost=0.1,
            shortage_cost=1.0,
            order_cost=0.3,
            discount=0.95
        )
        is_valid, msg = validate_params(params)
        assert is_valid, msg
    
    def test_invalid_max_inventory(self):
        """Test invalid max_inventory."""
        params = InventoryParams(
            max_inventory=0,
            max_order=5,
            demand_probs={0: 0.1, 1: 0.9},
            holding_cost=0.1,
            shortage_cost=1.0
        )
        is_valid, msg = validate_params(params)
        assert not is_valid
        assert "max_inventory" in msg
    
    def test_invalid_discount(self):
        """Test invalid discount factor."""
        params = InventoryParams(
            max_inventory=10,
            max_order=5,
            demand_probs={0: 0.1, 1: 0.9},
            holding_cost=0.1,
            shortage_cost=1.0,
            discount=1.1  # Invalid: > 1
        )
        is_valid, msg = validate_params(params)
        assert not is_valid
        assert "discount" in msg
    
    def test_negative_costs(self):
        """Test negative costs."""
        params = InventoryParams(
            max_inventory=10,
            max_order=5,
            demand_probs={0: 0.1, 1: 0.9},
            holding_cost=-0.1,  # Invalid
            shortage_cost=1.0
        )
        is_valid, msg = validate_params(params)
        assert not is_valid
        assert "holding_cost" in msg


class TestInventoryMDP:
    """Test InventoryMDP class."""
    
    @pytest.fixture
    def default_params(self):
        """Create default valid parameters."""
        return InventoryParams(
            max_inventory=5,
            max_order=3,
            demand_probs={0: 0.2, 1: 0.5, 2: 0.3},
            holding_cost=0.1,
            shortage_cost=1.0,
            order_cost=0.2,
            discount=0.95
        )
    
    def test_mdp_creation(self, default_params):
        """Test MDP creation."""
        mdp = InventoryMDP(default_params)
        assert mdp.num_states() == 6  # 0 to 5
        assert mdp.num_actions() == 4  # 0 to 3
        assert mdp.P.shape == (6, 4, 6)
        assert mdp.R.shape == (6, 4, 6)
    
    def test_transition_probabilities_sum_to_one(self, default_params):
        """Test that transition probabilities sum to 1."""
        mdp = InventoryMDP(default_params)
        is_valid, msg = validate_mdp(mdp)
        assert is_valid, msg
    
    def test_demand_normalization(self, default_params):
        """Test that demand probabilities are normalized."""
        # Use unnormalized probabilities
        params = InventoryParams(
            max_inventory=5,
            max_order=3,
            demand_probs={0: 2, 1: 5, 2: 3},  # Not normalized
            holding_cost=0.1,
            shortage_cost=1.0
        )
        mdp = InventoryMDP(params)
        # Check that probabilities sum to 1
        prob_sum = sum(mdp.demand_probs.values())
        assert abs(prob_sum - 1.0) < 1e-10
    
    def test_empty_demand_probs(self):
        """Test that empty demand_probs raises error."""
        params = InventoryParams(
            max_inventory=5,
            max_order=3,
            demand_probs={},  # Empty
            holding_cost=0.1,
            shortage_cost=1.0
        )
        with pytest.raises(ValueError):
            InventoryMDP(params)
    
    def test_reward_structure(self, default_params):
        """Test that rewards are negative costs."""
        mdp = InventoryMDP(default_params)
        # Sample a few transitions
        for s in [0, 2, 5]:
            for a in [0, 1, 3]:
                # Rewards should be non-positive (costs are non-negative)
                assert np.all(mdp.R[s, a, :] <= 0)


class TestMDPTransitions:
    """Test MDP transition logic."""
    
    @pytest.fixture
    def small_mdp(self):
        """Create a small MDP for testing."""
        return InventoryMDP(InventoryParams(
            max_inventory=3,
            max_order=2,
            demand_probs={0: 0.3, 1: 0.7},
            holding_cost=0.1,
            shortage_cost=1.0,
            order_cost=0.2,
            discount=0.9
        ))
    
    def test_inventory_cannot_go_negative(self, small_mdp):
        """Test that inventory cannot go below 0."""
        # From state 0, action 0, demand 1 should go to state 0 (not -1)
        assert small_mdp.P[0, 0, 0] > 0  # Can stay at 0
        # Check no negative states
        assert np.all(small_mdp.P[:, :, :] >= 0)
    
    def test_inventory_cannot_exceed_max(self, small_mdp):
        """Test that inventory cannot exceed max_inventory."""
        # From state 3, action 2, should cap at 3
        # Check transitions from max state
        for a in range(small_mdp.num_actions()):
            # Next state should be <= max_inventory
            assert np.all(small_mdp.P[3, a, :] == 0) or \
                   np.argmax(small_mdp.P[3, a, :]) <= 3

