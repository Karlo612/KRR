"""
config.py

Default parameter configuration for inventory MDP experiments.

This module provides a centralised location for default parameter settings used
in experiments. The default configuration represents a typical hospital drug
inventory scenario with reasonable cost parameters and demand distribution.
"""

from __future__ import annotations
from .mdp_inventory import InventoryParams


def default_params() -> InventoryParams:
    """
    Return default parameter configuration for inventory MDP experiments.
    
    This function provides a standard parameter set that can be used for
    initial experiments and demonstrations. The parameters represent a typical
    hospital inventory scenario with a moderate state space, realistic cost
    structure, and a demand distribution that captures typical variability.
    
    Returns
    -------
    InventoryParams
        Default parameter configuration with:
        - State space: 0-10 inventory units
        - Action space: 0-5 order quantities
        - Demand distribution: Poisson-like with mean around 2
        - Cost structure: holding (0.1), shortage (1.0), ordering (0.3)
        - Discount factor: 0.95
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