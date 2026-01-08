from __future__ import annotations

"""
mdp_inventory.py

Core finite-state MDP model for hospital drug inventory management.

This module implements a Markov Decision Process (MDP) for inventory control
in a hospital setting. The model captures the fundamental trade-off between
holding costs, shortage penalties, and ordering costs under stochastic demand,
enabling optimal ordering decisions through dynamic programming.

Mathematical Formulation:
    The inventory system is modelled as a finite-state discounted MDP with:
    
    - State space: S = {0, 1, 2, ..., max_inventory}
      Each state s represents the current inventory level (units in stock).
    
    - Action space: A = {0, 1, 2, ..., max_order}
      Each action a represents the order quantity for the current period.
    
    - Transition probabilities: P(s' | s, a)
      State transitions are determined by the stochastic demand distribution.
      After ordering a units in state s, demand d is realised, leading to
      next state s' = max(0, min(max_inventory, s + a) - d).
    
    - Reward function: R_sa(s, a) = expected immediate reward
      Computed as the negative expected cost: 
      R_sa(s, a) = Σ_d p(d) * (-(holding_cost * s' + shortage_cost * unmet + order_cost * a))
      where s' is the next state and unmet = max(0, d - (s + a)).
      This formulation converts cost minimisation into reward maximisation.
    
    - Objective: Find optimal policy π* that maximizes expected discounted reward:
      V^π(s) = E[Σ_{t=0}^∞ γ^t R_sa(s_t, a_t) | s_0 = s, π]

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
    Configuration parameters for the inventory MDP model.
    
    This dataclass encapsulates all parameters needed to define the inventory
    management problem, including state and action space bounds, demand
    distribution, cost structure, and discount factor.
    
    Attributes
    ----------
    max_inventory : int
        Maximum inventory level. The state space contains max_inventory + 1 states
        representing inventory levels from 0 to max_inventory. Must be >= 1.
    max_order : int
        Maximum order quantity per period. The action space contains max_order + 1
        actions representing order quantities from 0 to max_order. Must be >= 0.
    demand_probs : Dict[int, float]
        Discrete probability distribution over demand values. Keys are non-negative
        demand values, values are their corresponding probabilities. The distribution
        will be normalised automatically to ensure probabilities sum to unity.
    holding_cost : float
        Cost per unit of inventory held per period. Represents storage and
        opportunity costs. Must be >= 0.
    shortage_cost : float
        Penalty cost per unit of unmet demand per period. Represents the cost of
        stockouts. Must be >= 0.
    order_cost : float, optional
        Cost per unit ordered. Includes procurement and handling costs.
        Default is 0.0. Must be >= 0.
    discount : float, optional
        Discount factor γ for the infinite-horizon problem. Must be in (0, 1).
        Higher values place more weight on future rewards. Default is 0.95.
    """
    max_inventory: int
    max_order: int
    demand_probs: Dict[int, float]
    holding_cost: float
    shortage_cost: float
    order_cost: float = 0.0
    discount: float = 0.95
    
    def __repr__(self) -> str:
        """Return a readable string representation of the parameters."""
        return (
            f"InventoryParams(max_inventory={self.max_inventory}, "
            f"max_order={self.max_order}, "
            f"demand_probs={self.demand_probs}, "
            f"holding_cost={self.holding_cost}, "
            f"shortage_cost={self.shortage_cost}, "
            f"order_cost={self.order_cost}, "
            f"discount={self.discount})"
        )


class InventoryMDP:
    """
    Finite-state discounted MDP for inventory control.
    
    This class implements a hospital drug inventory management system as a
    Markov Decision Process. The MDP formulation enables optimal ordering
    decisions through dynamic programming algorithms like value iteration
    and policy iteration.
    
    The class precomputes the transition probability matrix P[s, a, s'] and
    expected reward matrix R_sa[s, a] during initialisation, which allows
    efficient computation of optimal policies using standard DP algorithms.
    
    Attributes
    ----------
    params : InventoryParams
        Configuration parameters for the MDP model.
    M : int
        Maximum inventory level (state space bound).
    K : int
        Maximum order quantity (action space bound).
    discount : float
        Discount factor γ for infinite-horizon discounted rewards.
    states : np.ndarray
        Array of all possible states [0, 1, ..., M].
    actions : np.ndarray
        Array of all possible actions [0, 1, ..., K].
    demand_probs : Dict[int, float]
        Normalised demand probability distribution (probabilities sum to 1.0).
    P : np.ndarray
        Transition probability matrix of shape (M+1, K+1, M+1).
        P[s, a, s'] gives the probability of transitioning from state s to s'
        when taking action a.
    R_sa : np.ndarray
        Expected immediate reward matrix of shape (M+1, K+1).
        R_sa[s, a] gives the expected immediate reward for taking action a
        in state s (negative expected cost).
    """

    def __init__(self, params: InventoryParams, validate: bool = True):
        """
        Initialise the inventory MDP from parameters.
        
        This constructor sets up the MDP by validating parameters (if requested),
        initialising state and action spaces, normalising the demand distribution,
        and precomputing the transition and reward matrices.
        
        Parameters
        ----------
        params : InventoryParams
            Configuration parameters for the MDP model.
        validate : bool, optional
            Whether to validate parameters before construction. Validation ensures
            all parameters are within valid ranges and constraints are satisfied.
            Default is True.
        
        Raises
        ------
        ValueError
            If parameters are invalid and validate=True, or if the demand
            distribution is empty or has invalid probabilities.
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

        # Normalise the demand probability distribution to ensure probabilities sum to unity
        self.demand_support = sorted(params.demand_probs.keys())
        if not self.demand_support:
            raise ValueError("demand_probs cannot be empty")
        
        probs = np.array([params.demand_probs[d] for d in self.demand_support], dtype=float)
        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("demand probabilities must sum to a positive value")
        probs /= prob_sum
        self.demand_probs = dict(zip(self.demand_support, probs))

        # Allocate matrices for transition probabilities and expected rewards
        # P[s, a, s'] stores the probability of transitioning from state s to s' under action a
        # R_sa[s, a] stores the expected immediate reward for taking action a in state s
        self.P = np.zeros((self.M + 1, self.K + 1, self.M + 1), dtype=float)
        self.R_sa = np.zeros((self.M + 1, self.K + 1), dtype=float)

        self._build_transition_and_reward_matrices()

    def _build_transition_and_reward_matrices(self):
        """
        Build the transition probability matrix P and expected reward matrix R_sa.
        
        This method constructs the core MDP matrices by enumerating all possible
        state transitions and computing their probabilities and associated rewards.
        For each state-action pair (s, a), we consider all possible demand
        realisations and compute the resulting next state and costs.
        
        The algorithm proceeds as follows:
        1. For each state s and action a, compute post-order inventory level
        2. For each possible demand value d (weighted by probability p_d):
           - Determine next inventory state (after demand is realised)
           - Compute unmet demand (shortages)
           - Calculate total cost (holding + shortage + ordering)
           - Accumulate transition probabilities and expected rewards
        3. Normalise transition probabilities so each (s, a) row sums to 1.0
        
        The reward is defined as the negative cost, converting the cost
        minimisation problem into a reward maximisation problem suitable for
        standard MDP solution algorithms.
        """
        M = self.M
        K = self.K

        hp = self.params.holding_cost
        sp = self.params.shortage_cost
        oc = self.params.order_cost

        for s in range(M + 1):
            for a in range(K + 1):
                # Compute inventory level after order is placed (capped at maximum)
                inv_pre = min(M, s + a)

                # Consider each possible demand realisation and its probability
                for d, p_d in self.demand_probs.items():
                    # Compute next period inventory (cannot go below zero)
                    next_inv = max(0, inv_pre - d)
                    
                    # Calculate unmet demand (shortages when demand exceeds available inventory)
                    unmet = max(0, d - inv_pre)

                    # Compute total cost as sum of holding, shortage, and ordering costs
                    total_cost = hp * next_inv + sp * unmet + oc * a
                    
                    # Reward is negative cost (cost minimisation → reward maximisation)
                    reward = -total_cost

                    # Accumulate transition probability for this state transition
                    self.P[s, a, next_inv] += p_d
                    
                    # Accumulate expected reward (weighted by demand probability)
                    self.R_sa[s, a] += p_d * reward

        # Normalise transition probabilities so each (s, a) pair sums to 1.0
        # This ensures P represents a valid probability distribution over next states
        row_sums = self.P.sum(axis=2, keepdims=True)
        self.P = np.divide(
            self.P,
            np.where(row_sums == 0, 1, row_sums),
            out=self.P,
            where=row_sums != 0
        )

    def num_states(self) -> int:
        """
        Return the number of states in the MDP state space.
        
        Returns
        -------
        int
            Number of states, equal to max_inventory + 1 (states 0 through M).
        """
        return self.M + 1

    def num_actions(self) -> int:
        """
        Return the number of actions in the MDP action space.
        
        Returns
        -------
        int
            Number of actions, equal to max_order + 1 (actions 0 through K).
        """
        return self.K + 1