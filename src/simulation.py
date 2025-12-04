"""
simulation.py

Simulation engine for an inventory-control MDP.

This extended version provides:
- full cost breakdown (holding, shortage, ordering)
- per-step cost trajectory
- per-step reward trajectory
- inventory trajectory
- demand trajectory
- service-level statistics
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

    Parameters
    ----------
    mdp : InventoryMDP
    policy_fn : callable
        policy_fn(s) -> action a
    T : int
        Number of time steps.
    initial_state : int
        Starting inventory state.
    seed : int or None
        Random seed.

    Returns
    -------
    metrics : dict
        Contains:
        - total_reward
        - total_cost
        - avg_cost_per_period
        - total_demand
        - fulfilled_demand
        - unmet_demand
        - service_level
        - per_step_costs
        - per_step_holding
        - per_step_shortage
        - per_step_ordering
        - rewards
        - inventory_levels
        - demand_sequence
        - actions
        - T
    """
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