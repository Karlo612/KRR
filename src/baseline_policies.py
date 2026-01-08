"""
baseline_policies.py

Heuristic inventory policies for baseline comparison.

This module implements common heuristic inventory management policies that serve
as baselines for evaluating the performance of MDP-derived optimal policies.
These heuristics are widely used in practice and provide a benchmark against
which to compare the optimal dynamic programming solutions.
"""

from __future__ import annotations
from typing import Callable


def make_sS_policy(s: int, S: int) -> Callable[[int], int]:
    """
    Create an (s, S) policy function (also known as a two-bin policy).

    The (s, S) policy is a threshold-based heuristic that is widely used in
    inventory management. When the current stock level falls at or below the
    reorder point s, the policy orders enough to bring inventory up to the
    order-up-to level S. Otherwise, no order is placed.

    Parameters
    ----------
    s : int
        Reorder point (threshold). When inventory falls to s or below, an order
        is triggered. Must be >= 0.
    S : int
        Order-up-to level. The policy orders enough to reach this level when
        inventory is at or below s. Must be >= s.

    Returns
    -------
    policy_fn : Callable[[int], int]
        Policy function that maps inventory level (state) to order quantity
        (action). The function takes a state and returns an action.

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
            # Order enough to bring inventory up to level S
            return max(S - stock, 0)
        return 0

    return policy_fn


def fixed_order_policy(order_quantity: int) -> Callable[[int], int]:
    """
    Create a fixed-order quantity policy function.

    This simple heuristic policy always orders the same fixed quantity,
    regardless of the current inventory level. While not adaptive, it provides
    a straightforward baseline for comparison with more sophisticated policies.

    Parameters
    ----------
    order_quantity : int
        Fixed order quantity to place in every period. Must be >= 0.

    Returns
    -------
    policy_fn : Callable[[int], int]
        Policy function that always returns the fixed order quantity,
        regardless of the input state.

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
    Create a reorder-to-level policy function.

    This policy orders enough inventory to reach a target level whenever the
    current inventory is below that target. If inventory is already at or
    above the target, no order is placed. This is a simple threshold policy
    that maintains inventory around a desired level.

    Parameters
    ----------
    target_level : int
        Target inventory level to maintain. The policy orders to reach this
        level when inventory is below it. Must be >= 0.

    Returns
    -------
    policy_fn : Callable[[int], int]
        Policy function that orders enough to reach the target level when
        inventory is below it, and orders nothing otherwise.

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