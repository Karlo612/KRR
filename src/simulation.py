"""
simulation.py

Simulation engine for inventory-control MDP policies.

This module provides functionality to simulate the inventory system dynamics
under a given policy over a specified time horizon. The simulation generates
trajectories of inventory levels, demands, and actions, and computes
comprehensive performance metrics for policy evaluation.

The simulation tracks various metrics including:
- Total and average costs over the time horizon
- Cost component breakdown (holding, shortage, ordering costs)
- Per-step cost and reward trajectories
- Inventory and demand time series
- Service level (demand fulfilment rate)
- Action sequences over time
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
    Simulate the inventory system under a given policy for T time periods.

    This function simulates the inventory dynamics by following the policy at
    each time step. For each period, the simulation:
    1. Observes the current inventory state s_t
    2. Selects an action a_t = policy_fn(s_t) according to the policy
    3. Updates inventory after ordering: inv_pre = min(max_inventory, s_t + a_t)
    4. Samples demand d_t from the demand distribution
    5. Updates inventory after demand: s_{t+1} = max(0, inv_pre - d_t)
    6. Computes costs (holding, shortage, ordering) for the period
    7. Records all metrics for analysis

    The simulation uses a random number generator with an optional seed for
    reproducibility, enabling fair comparisons between different policies using
    identical demand sequences.

    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance defining the inventory system dynamics, including state
        and action spaces, demand distribution, and cost parameters.
    policy_fn : Callable[[int], int]
        Policy function that maps inventory state to order quantity. The function
        must satisfy: 0 <= policy_fn(s) <= max_order for all valid states s.
    T : int, optional
        Number of time steps (periods) to simulate. Must be > 0. Default is 200.
    initial_state : int, optional
        Starting inventory state. Must satisfy: 0 <= initial_state <= max_inventory.
        Default is 0 (empty inventory).
    seed : int | None, optional
        Random seed for the demand sampling process. If provided, ensures
        reproducible simulations. If None, uses a non-deterministic seed.
        Default is None.

    Returns
    -------
    metrics : Dict[str, np.ndarray | float | int]
        Dictionary containing comprehensive performance metrics:
        - total_reward (float): Sum of rewards over T periods
        - total_cost (float): Sum of costs over T periods
        - avg_cost_per_period (float): Average cost per period
        - total_demand (int): Total demand realised over T periods
        - fulfilled_demand (int): Total demand that was satisfied from inventory
        - unmet_demand (int): Total demand that could not be satisfied (shortages)
        - service_level (float): Demand fulfilment rate (fulfilled/total)
        - per_step_costs (np.ndarray): Cost at each time step, shape (T,)
        - per_step_holding (np.ndarray): Holding cost at each step, shape (T,)
        - per_step_shortage (np.ndarray): Shortage cost at each step, shape (T,)
        - per_step_ordering (np.ndarray): Ordering cost at each step, shape (T,)
        - rewards (np.ndarray): Reward at each step, shape (T,)
        - inventory_levels (np.ndarray): Inventory level at each step, shape (T+1,)
        - demand_sequence (np.ndarray): Demand realised at each step, shape (T,)
        - actions (np.ndarray): Action taken at each step, shape (T,)
        - T (int): Number of time steps simulated

    Raises
    ------
    ValueError
        If initial_state is outside the valid range or T <= 0.

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
    # Validate input parameters
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if initial_state < 0 or initial_state > mdp.params.max_inventory:
        raise ValueError(
            f"initial_state must be in [0, {mdp.params.max_inventory}], "
            f"got {initial_state}"
        )
    
    # Initialise random number generator for demand sampling
    # Using a seed ensures reproducible simulations for fair policy comparisons
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    params = mdp.params

    # Allocate arrays to store simulation trajectories and metrics
    inventory_levels = np.zeros(T+1, dtype=int)
    demand_sequence = np.zeros(T, dtype=int)
    actions = np.zeros(T, dtype=int)

    per_step_costs = np.zeros(T, dtype=float)
    per_step_holding = np.zeros(T, dtype=float)
    per_step_shortage = np.zeros(T, dtype=float)
    per_step_ordering = np.zeros(T, dtype=float)
    rewards = np.zeros(T, dtype=float)

    # Set initial inventory state
    inventory_levels[0] = initial_state

    # Extract demand values and probabilities for efficient sampling
    demand_vals = np.array(list(mdp.demand_probs.keys()))
    demand_probs = np.array(list(mdp.demand_probs.values()))

    # Initialise aggregators for service level computation
    total_demand = 0
    fulfilled_total = 0
    unmet_total = 0

    for t in range(T):
        # Get current inventory state and select action according to policy
        s = inventory_levels[t]
        a = policy_fn(s)
        actions[t] = a

        # Apply order: inventory increases by order quantity, capped at maximum
        inv_pre = min(params.max_inventory, s + a)

        # Sample demand from the demand distribution
        demand = rng.choice(demand_vals, p=demand_probs)
        demand_sequence[t] = demand

        # Update inventory after demand is realised (cannot go below zero)
        next_inv = max(0, inv_pre - demand)
        inventory_levels[t+1] = next_inv

        # Compute fulfilled and unmet demand for service level calculation
        fulfilled = min(inv_pre, demand)
        unmet = max(0, demand - inv_pre)

        # Compute cost components for this period
        holding_cost = params.holding_cost * next_inv
        shortage_cost = params.shortage_cost * unmet
        ordering_cost = params.order_cost * a

        total_cost = holding_cost + shortage_cost + ordering_cost
        reward = -total_cost

        # Record per-step metrics for detailed analysis
        per_step_costs[t] = total_cost
        per_step_holding[t] = holding_cost
        per_step_shortage[t] = shortage_cost
        per_step_ordering[t] = ordering_cost
        rewards[t] = reward

        # Accumulate totals for aggregate metrics (service level)
        total_demand += demand
        fulfilled_total += fulfilled
        unmet_total += unmet

    # Compute aggregate metrics over the simulation horizon
    total_cost = per_step_costs.sum()
    total_reward = rewards.sum()
    avg_cost = total_cost / T
    # Service level: proportion of demand that was fulfilled
    # If no demand occurs, service level is defined as 1.0 (all demand fulfilled)
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