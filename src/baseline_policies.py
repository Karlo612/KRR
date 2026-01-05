"""
baseline_policies.py

Heuristic inventory policies for comparison with optimal MDP policies.

These policies serve as baselines to evaluate the performance of MDP-derived
optimal policies. They implement common inventory management heuristics.
"""

from __future__ import annotations
from typing import Callable


def make_sS_policy(s: int, S: int) -> Callable[[int], int]:
    """
    Factory for an (s, S) policy (also known as two-bin policy).

    The (s, S) policy is a threshold-based heuristic:
    - If current stock level x ≤ s (reorder point), order up to S (order-up-to level)
    - Otherwise, order 0

    This is a widely used heuristic in inventory management.

    Parameters
    ----------
    s : int
        Reorder point (threshold). Must be >= 0.
    S : int
        Order-up-to level. Must be >= s.

    Returns
    -------
    policy_fn : Callable[[int], int]
        Policy function that maps state (inventory level) to action (order quantity).
        policy_fn(state) -> action (int)

    Examples
    --------
    >>> policy = make_sS_policy(s=2, S=5)
    >>> policy(0)  # stock=0 <= s=2, order up to 5
    5
    >>> policy(2)  # stock=2 <= s=2, order up to 5
    3
    >>> policy(3)  # stock=3 > s=2, order 0
    0
    """
    if s < 0:
        raise ValueError(f"Reorder point s must be >= 0, got {s}")
    if S < s:
        raise ValueError(f"Order-up-to level S must be >= s, got S={S}, s={s}")
    
    def policy_fn(stock: int) -> int:
        if stock < 0:
            raise ValueError(f"Stock level must be >= 0, got {stock}")
        if stock <= s:
            # Order quantity to reach S
            return max(S - stock, 0)
        return 0

    return policy_fn


def fixed_order_policy(order_quantity: int) -> Callable[[int], int]:
    """
    Factory for a fixed-order quantity policy.

    Always orders a fixed quantity, regardless of current inventory level.
    This is a simple heuristic for comparison.

    Parameters
    ----------
    order_quantity : int
        Fixed order quantity. Must be >= 0.

    Returns
    -------
    policy_fn : Callable[[int], int]
        Policy function that always returns order_quantity.
        policy_fn(state) -> action (int)

    Examples
    --------
    >>> policy = fixed_order_policy(order_quantity=3)
    >>> policy(0)
    3
    >>> policy(10)
    3
    """
    if order_quantity < 0:
        raise ValueError(f"Order quantity must be >= 0, got {order_quantity}")
    
    def policy_fn(stock: int) -> int:
        if stock < 0:
            raise ValueError(f"Stock level must be >= 0, got {stock}")
        return order_quantity

    return policy_fn


def reorder_to_level_policy(target_level: int) -> Callable[[int], int]:
    """
    Factory for a reorder-to-level policy.

    Always orders enough to reach a target inventory level, if current
    inventory is below the target.

    Parameters
    ----------
    target_level : int
        Target inventory level. Must be >= 0.

    Returns
    -------
    policy_fn : Callable[[int], int]
        Policy function that orders to reach target_level.
        policy_fn(state) -> action (int)

    Examples
    --------
    >>> policy = reorder_to_level_policy(target_level=5)
    >>> policy(2)  # order 3 to reach 5
    3
    >>> policy(5)  # already at target, order 0
    0
    >>> policy(7)  # above target, order 0
    0
    """
    if target_level < 0:
        raise ValueError(f"Target level must be >= 0, got {target_level}")
    
    def policy_fn(stock: int) -> int:
        if stock < 0:
            raise ValueError(f"Stock level must be >= 0, got {stock}")
        if stock < target_level:
            return target_level - stock
        return 0

    return policy_fn