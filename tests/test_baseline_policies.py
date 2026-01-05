"""
Unit tests for baseline_policies module.
"""

import pytest
from src.baseline_policies import (
    make_sS_policy,
    fixed_order_policy,
    reorder_to_level_policy
)


class TestSSPolicy:
    """Test (s, S) policy."""
    
    def test_basic_functionality(self):
        """Test basic (s, S) policy behavior."""
        policy = make_sS_policy(s=2, S=5)
        assert policy(0) == 5  # Order up to 5
        assert policy(1) == 4  # Order 4 to reach 5
        assert policy(2) == 3  # Order 3 to reach 5
        assert policy(3) == 0  # Above threshold, order 0
        assert policy(5) == 0  # Above threshold, order 0
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            make_sS_policy(s=-1, S=5)  # Negative s
        
        with pytest.raises(ValueError):
            make_sS_policy(s=5, S=3)  # S < s
    
    def test_invalid_state(self):
        """Test that invalid state raises error."""
        policy = make_sS_policy(s=2, S=5)
        with pytest.raises(ValueError):
            policy(-1)  # Negative state


class TestFixedOrderPolicy:
    """Test fixed order policy."""
    
    def test_basic_functionality(self):
        """Test fixed order policy behavior."""
        policy = fixed_order_policy(order_quantity=3)
        assert policy(0) == 3
        assert policy(5) == 3
        assert policy(10) == 3
    
    def test_invalid_quantity(self):
        """Test that negative quantity raises error."""
        with pytest.raises(ValueError):
            fixed_order_policy(order_quantity=-1)


class TestReorderToLevelPolicy:
    """Test reorder-to-level policy."""
    
    def test_basic_functionality(self):
        """Test reorder-to-level policy behavior."""
        policy = reorder_to_level_policy(target_level=5)
        assert policy(0) == 5  # Order 5
        assert policy(2) == 3  # Order 3 to reach 5
        assert policy(5) == 0  # At target, order 0
        assert policy(7) == 0  # Above target, order 0
    
    def test_invalid_target(self):
        """Test that negative target raises error."""
        with pytest.raises(ValueError):
            reorder_to_level_policy(target_level=-1)

