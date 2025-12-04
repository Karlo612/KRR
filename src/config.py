"""
config.py

Central place for default InventoryMDP parameter settings.
"""

from __future__ import annotations
from .mdp_inventory import InventoryParams


def default_params() -> InventoryParams:
    """
    Default configuration for experiments.
    """
    return InventoryParams(
        max_inventory=10,
        max_order=5,
        demand_probs={0: 0.1, 1: 0.3, 2: 0.3, 3: 0.2, 4: 0.1},
        holding_cost=0.1,
        shortage_cost=1.0,
        order_cost=0.3,
        discount=0.95
    )