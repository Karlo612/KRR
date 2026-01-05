from __future__ import annotations

"""
mdp_inventory.py

Core finite-state MDP model for a hospital drug inventory system.

This module implements a Markov Decision Process (MDP) for inventory control
in a hospital setting. The MDP models the trade-off between holding costs,
shortage penalties, and ordering costs under stochastic demand.

Mathematical Formulation:
    - State space: S = {0, 1, 2, ..., max_inventory}
      Each state represents current inventory level.
    
    - Action space: A = {0, 1, 2, ..., max_order}
      Each action represents order quantity.
    
    - Transition probabilities: P(s' | s, a)
      Determined by stochastic demand distribution.
    
    - Reward function: R(s, a, s') = -(holding_cost * s' + shortage_cost * unmet + order_cost * a)
      Negative of total cost (cost minimization problem).
    
    - Objective: Find policy π* that maximizes expected discounted reward:
      V^π(s) = E[Σ_{t=0}^∞ γ^t R(s_t, a_t, s_{t+1}) | s_0 = s, π]

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
    """
    Parameters for the inventory MDP model.
    
    Attributes
    ----------
    max_inventory : int
        Maximum inventory level (state space size = max_inventory + 1).
        Must be >= 1.
    max_order : int
        Maximum order quantity (action space size = max_order + 1).
        Must be >= 0.
    demand_probs : Dict[int, float]
        Discrete demand distribution. Keys are demand values (non-negative),
        values are probabilities (will be normalized to sum to 1).
    holding_cost : float
        Per-unit holding cost per period. Must be >= 0.
    shortage_cost : float
        Per-unit shortage penalty per period. Must be >= 0.
    order_cost : float, optional
        Per-unit ordering cost. Default is 0.0. Must be >= 0.
    discount : float, optional
        Discount factor γ ∈ (0, 1) for infinite horizon. Default is 0.95.
    """
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
    
    This class implements a hospital drug inventory management system as an MDP.
    It precomputes transition probability matrix P[s, a, s'] and expected reward
    matrix R[s, a, s'] for efficient dynamic programming algorithms.
    
    Attributes
    ----------
    params : InventoryParams
        MDP parameters.
    M : int
        Maximum inventory level.
    K : int
        Maximum order quantity.
    discount : float
        Discount factor γ.
    states : np.ndarray
        State space array [0, 1, ..., M].
    actions : np.ndarray
        Action space array [0, 1, ..., K].
    demand_probs : Dict[int, float]
        Normalized demand probability distribution.
    P : np.ndarray
        Transition probability matrix, shape (M+1, K+1, M+1).
        P[s, a, s'] = probability of transitioning from state s to s' under action a.
    R : np.ndarray
        Expected reward matrix, shape (M+1, K+1, M+1).
        R[s, a, s'] = expected immediate reward for transition (s, a) → s'.
    """

    def __init__(self, params: InventoryParams, validate: bool = True):
        """
        Initialize the inventory MDP.
        
        Parameters
        ----------
        params : InventoryParams
            MDP parameters.
        validate : bool, optional
            If True, validate parameters before construction. Default is True.
        
        Raises
        ------
        ValueError
            If parameters are invalid and validate=True.
        """
        if validate:
            from .validation import validate_params
            is_valid, error_msg = validate_params(params)
            if not is_valid:
                raise ValueError(f"Invalid parameters: {error_msg}")
        
        self.params = params
        self.M = params.max_inventory
        self.K = params.max_order
        self.discount = params.discount

        self.states = np.arange(self.M + 1)
        self.actions = np.arange(self.K + 1)

        # Normalise demand distribution
        self.demand_support = sorted(params.demand_probs.keys())
        if not self.demand_support:
            raise ValueError("demand_probs cannot be empty")
        
        probs = np.array([params.demand_probs[d] for d in self.demand_support], dtype=float)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("demand probabilities must sum to a positive value")
        probs /= prob_sum
        self.demand_probs = dict(zip(self.demand_support, probs))

        # Allocate P and R matrices
        # P[s, a, s']: probability of transition from state s to s' under action a
        # R[s, a, s']: expected immediate reward for transition (s, a) → s'
        self.P = np.zeros((self.M + 1, self.K + 1, self.M + 1), dtype=float)
        self.R = np.zeros((self.M + 1, self.K + 1, self.M + 1), dtype=float)

        self._build_transition_and_reward_matrices()

    def _build_transition_and_reward_matrices(self):
        """
        Build transition probability matrix P and expected reward matrix R.
        
        For each state s and action a:
        1. Compute post-order inventory: inv_pre = min(M, s + a)
        2. For each possible demand d with probability p_d:
           - Compute next inventory: next_inv = max(0, inv_pre - d)
           - Compute unmet demand: unmet = max(0, d - inv_pre)
           - Compute cost: holding_cost * next_inv + shortage_cost * unmet + order_cost * a
           - Reward = -cost (cost minimization → reward maximization)
           - Accumulate: P[s, a, next_inv] += p_d
           - Accumulate: R[s, a, next_inv] += p_d * reward
        
        After accumulation, normalize P so each row sums to 1.0.
        """
        M = self.M
        K = self.K

        hp = self.params.holding_cost
        sp = self.params.shortage_cost
        oc = self.params.order_cost

        for s in range(M + 1):
            for a in range(K + 1):
                # Post-order inventory (capped at max_inventory)
                inv_pre = min(M, s + a)

                # For each possible demand realization
                for d, p_d in self.demand_probs.items():
                    # Next period inventory (cannot go negative)
                    next_inv = max(0, inv_pre - d)
                    
                    # Unmet demand (shortage)
                    unmet = max(0, d - inv_pre)

                    # Total cost components
                    total_cost = hp * next_inv + sp * unmet + oc * a
                    
                    # Reward is negative cost (minimize cost = maximize reward)
                    reward = -total_cost

                    # Accumulate transition probability
                    self.P[s, a, next_inv] += p_d
                    
                    # Accumulate expected reward (weighted by probability)
                    self.R[s, a, next_inv] += p_d * reward

        # Normalize transition probabilities to sum to 1.0 for each (s, a) pair
        row_sums = self.P.sum(axis=2, keepdims=True)
        self.P = np.divide(
            self.P,
            np.where(row_sums == 0, 1, row_sums),
            out=self.P,
            where=row_sums != 0
        )

    def num_states(self) -> int:
        """
        Return the number of states in the MDP.
        
        Returns
        -------
        int
            Number of states (max_inventory + 1).
        """
        return self.M + 1

    def num_actions(self) -> int:
        """
        Return the number of actions in the MDP.
        
        Returns
        -------
        int
            Number of actions (max_order + 1).
        """
        return self.K + 1