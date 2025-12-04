"""
baseline_policies.py

Heuristic inventory policies for comparison with optimal MDP policies.

Includes:
- (s, S) policy: if stock ≤ s, order up to S; else order 0.
"""

from __future__ import annotations


def make_sS_policy(s: int, S: int):
    """
    Factory for an (s, S) policy.

    If current stock level x ≤ s, order (S - x), else order 0.

    Returns
    -------
    policy_fn : callable
        policy_fn(state) -> action (int)
    """
    def policy_fn(stock: int) -> int:
        if stock <= s:
            # quantity required to reach S
            return max(S - stock, 0)
        return 0

    return policy_fn