"""
simulation.py

Simulation engine for an inventory-control MDP.

This module provides functionality to simulate inventory system dynamics
under a given policy and collect comprehensive performance metrics.

Metrics provided:
- Total and average costs
- Cost component breakdown (holding, shortage, ordering)
- Per-step cost and reward trajectories
- Inventory and demand trajectories
- Service level (fulfillment rate)
- Action sequences
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict

from .mdp_inventory import InventoryMDP


def simulate_policy(
    mdp: InventoryMDP,
    policy_fn: Callable[[int], int],
    T: int = 200,
    initial_state: int = 0,
    seed: int | None = None,
) -> Dict[str, np.ndarray | float | int]:
    """
    Simulate T periods of the inventory system using the provided policy.

    The simulation follows this process for each time step:
    1. Observe current inventory state s_t
    2. Select action a_t = policy_fn(s_t)
    3. Update inventory: inv_pre = min(max_inventory, s_t + a_t)
    4. Sample demand d_t from demand distribution
    5. Update inventory: s_{t+1} = max(0, inv_pre - d_t)
    6. Compute costs: holding, shortage, ordering
    7. Record metrics

    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance defining the inventory system dynamics.
    policy_fn : Callable[[int], int]
        Policy function mapping state to action.
        Must satisfy: 0 <= policy_fn(s) <= max_order for all valid states s.
    T : int, optional
        Number of time steps to simulate. Default is 200.
        Must be > 0.
    initial_state : int, optional
        Starting inventory state. Default is 0.
        Must satisfy: 0 <= initial_state <= max_inventory.
    seed : int | None, optional
        Random seed for reproducibility. Default is None (non-deterministic).

    Returns
    -------
    metrics : Dict[str, np.ndarray | float | int]
        Dictionary containing:
        - total_reward (float): Sum of rewards over T periods
        - total_cost (float): Sum of costs over T periods
        - avg_cost_per_period (float): Average cost per period
        - total_demand (int): Total demand over T periods
        - fulfilled_demand (int): Total fulfilled demand
        - unmet_demand (int): Total unmet demand (shortages)
        - service_level (float): Fulfillment rate (fulfilled/total)
        - per_step_costs (np.ndarray): Cost at each time step, shape (T,)
        - per_step_holding (np.ndarray): Holding cost at each step, shape (T,)
        - per_step_shortage (np.ndarray): Shortage cost at each step, shape (T,)
        - per_step_ordering (np.ndarray): Ordering cost at each step, shape (T,)
        - rewards (np.ndarray): Reward at each step, shape (T,)
        - inventory_levels (np.ndarray): Inventory at each step, shape (T+1,)
        - demand_sequence (np.ndarray): Demand at each step, shape (T,)
        - actions (np.ndarray): Action at each step, shape (T,)
        - T (int): Number of time steps

    Raises
    ------
    ValueError
        If initial_state is invalid or T <= 0.

    Examples
    --------
    >>> from src.config import default_params
    >>> from src.mdp_inventory import InventoryMDP
    >>> from src.solvers import value_iteration
    >>> 
    >>> params = default_params()
    >>> mdp = InventoryMDP(params)
    >>> V, policy, _ = value_iteration(mdp)
    >>> 
    >>> def policy_fn(s): return int(policy[s])
    >>> metrics = simulate_policy(mdp, policy_fn, T=100, seed=42)
    >>> print(f"Average cost: {metrics['avg_cost_per_period']:.2f}")
    """
    # Validate inputs
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if initial_state < 0 or initial_state > mdp.params.max_inventory:
        raise ValueError(
            f"initial_state must be in [0, {mdp.params.max_inventory}], "
            f"got {initial_state}"
        )
    
    # Initialize random number generator
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    params = mdp.params

    # Storage arrays
    inventory_levels = np.zeros(T+1, dtype=int)
    demand_sequence = np.zeros(T, dtype=int)
    actions = np.zeros(T, dtype=int)

    per_step_costs = np.zeros(T, dtype=float)
    per_step_holding = np.zeros(T, dtype=float)
    per_step_shortage = np.zeros(T, dtype=float)
    per_step_ordering = np.zeros(T, dtype=float)
    rewards = np.zeros(T, dtype=float)

    # Initialise
    inventory_levels[0] = initial_state

    # Precompute demand distribution
    demand_vals = np.array(list(mdp.demand_probs.keys()))
    demand_probs = np.array(list(mdp.demand_probs.values()))

    total_demand = 0
    fulfilled_total = 0
    unmet_total = 0

    for t in range(T):
        s = inventory_levels[t]
        a = policy_fn(s)
        actions[t] = a

        # Apply order
        inv_pre = min(params.max_inventory, s + a)

        # Sample demand
        demand = rng.choice(demand_vals, p=demand_probs)
        demand_sequence[t] = demand

        # Update inventory
        next_inv = max(0, inv_pre - demand)
        inventory_levels[t+1] = next_inv

        fulfilled = min(inv_pre, demand)
        unmet = max(0, demand - inv_pre)

        # Cost components
        holding_cost = params.holding_cost * next_inv
        shortage_cost = params.shortage_cost * unmet
        ordering_cost = params.order_cost * a

        total_cost = holding_cost + shortage_cost + ordering_cost
        reward = -total_cost

        # Save per-step metrics
        per_step_costs[t] = total_cost
        per_step_holding[t] = holding_cost
        per_step_shortage[t] = shortage_cost
        per_step_ordering[t] = ordering_cost
        rewards[t] = reward

        # Aggregates for service-level
        total_demand += demand
        fulfilled_total += fulfilled
        unmet_total += unmet

    # Compute final metrics
    total_cost = per_step_costs.sum()
    total_reward = rewards.sum()
    avg_cost = total_cost / T
    service_level = fulfilled_total / max(total_demand, 1)

    return {
        "total_reward": total_reward,
        "total_cost": total_cost,
        "avg_cost_per_period": avg_cost,
        "total_demand": total_demand,
        "fulfilled_demand": fulfilled_total,
        "unmet_demand": unmet_total,
        "service_level": service_level,
        "per_step_costs": per_step_costs,
        "per_step_holding": per_step_holding,
        "per_step_shortage": per_step_shortage,
        "per_step_ordering": per_step_ordering,
        "rewards": rewards,
        "inventory_levels": inventory_levels,
        "demand_sequence": demand_sequence,
        "actions": actions,
        "T": T,
    }