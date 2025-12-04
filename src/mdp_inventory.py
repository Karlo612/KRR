from __future__ import annotations

"""
mdp_inventory.py

Core finite-state MDP model for a hospital drug inventory system.

States:
    s ∈ {0, 1, ..., max_inventory}  – current stock level

Actions:
    a ∈ {0, 1, ..., max_order}      – order quantity this period
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class InventoryParams:
    max_inventory: int
    max_order: int
    demand_probs: Dict[int, float]
    holding_cost: float
    shortage_cost: float
    order_cost: float = 0.0
    discount: float = 0.95


class InventoryMDP:
    """
    Finite-state discounted MDP for inventory control.
    """

    def __init__(self, params: InventoryParams):
        self.params = params
        self.M = params.max_inventory
        self.K = params.max_order
        self.discount = params.discount

        self.states = np.arange(self.M + 1)
        self.actions = np.arange(self.K + 1)

        # Normalise demand distribution
        self.demand_support = sorted(params.demand_probs.keys())
        probs = np.array([params.demand_probs[d] for d in self.demand_support], dtype=float)
        probs /= probs.sum()
        self.demand_probs = dict(zip(self.demand_support, probs))

        # Allocate P and R
        self.P = np.zeros((self.M + 1, self.K + 1, self.M + 1), float)
        self.R = np.zeros((self.M + 1, self.K + 1, self.M + 1), float)

        self._build_transition_and_reward_matrices()

    def _build_transition_and_reward_matrices(self):
        M = self.M
        K = self.K

        hp = self.params.holding_cost
        sp = self.params.shortage_cost
        oc = self.params.order_cost

        for s in range(M + 1):
            for a in range(K + 1):
                inv_pre = min(M, s + a)

                for d, p_d in self.demand_probs.items():
                    next_inv = max(0, inv_pre - d)
                    unmet = max(0, d - inv_pre)

                    total_cost = hp * next_inv + sp * unmet + oc * a
                    reward = - total_cost

                    self.P[s, a, next_inv] += p_d
                    self.R[s, a, next_inv] += p_d * reward

        row_sums = self.P.sum(axis=2, keepdims=True)
        self.P = np.divide(
            self.P,
            np.where(row_sums == 0, 1, row_sums),
            out=self.P,
            where=row_sums != 0
        )

    def num_states(self) -> int:
        return self.M + 1

    def num_actions(self) -> int:
        return self.K + 1