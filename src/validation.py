"""
validation.py

Validation functions for MDP parameters and structure.

This module provides validation functions to ensure that MDP parameters are
within valid ranges and that the MDP structure (e.g., transition probabilities)
is well-formed. Validation helps catch configuration errors early and ensures
the MDP model is mathematically consistent.
"""

from __future__ import annotations
import numpy as np
from .mdp_inventory import InventoryParams, InventoryMDP


def validate_params(params: InventoryParams) -> tuple[bool, str]:
    """
    Validate inventory MDP parameters for consistency and validity.
    
    This function checks that all parameters are within their valid ranges and
    that constraints are satisfied. It returns a boolean indicating validity
    and an error message describing any issues found.
    
    Parameters
    ----------
    params : InventoryParams
        Parameter configuration to validate.
    
    Returns
    -------
    is_valid : bool
        True if all parameters are valid, False otherwise.
    error_msg : str
        Empty string if valid, otherwise a concatenated error message
        describing all validation failures.
    """
    errors = []
    
    # Validate state space bounds
    if params.max_inventory < 1:
        errors.append("max_inventory must be >= 1")
    
    # Validate action space bounds
    if params.max_order < 0:
        errors.append("max_order must be >= 0")
    
    # Validate demand distribution
    if not params.demand_probs:
        errors.append("demand_probs cannot be empty")
    
    if any(d < 0 for d in params.demand_probs.keys()):
        errors.append("demand values must be non-negative")
    
    if any(p < 0 for p in params.demand_probs.values()):
        errors.append("demand probabilities must be non-negative")
    
    # Validate cost parameters (all must be non-negative)
    if params.holding_cost < 0:
        errors.append("holding_cost must be non-negative")
    
    if params.shortage_cost < 0:
        errors.append("shortage_cost must be non-negative")
    
    if params.order_cost < 0:
        errors.append("order_cost must be non-negative")
    
    # Validate discount factor (must be in open interval (0, 1))
    if params.discount <= 0 or params.discount >= 1:
        errors.append("discount must be in (0, 1)")
    
    if errors:
        return False, "; ".join(errors)
    return True, ""


def validate_mdp(mdp: InventoryMDP, tol: float = 1e-6) -> tuple[bool, str]:
    """
    Validate MDP structure by checking transition probabilities.
    
    This function verifies that the transition probability matrix is well-formed
    by ensuring that for each state-action pair, the probabilities over next
    states sum to 1.0 (within tolerance). This is a fundamental requirement for
    a valid MDP model.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance to validate.
    tol : float
        Numerical tolerance for probability sum checks. Default is 1e-6.
    
    Returns
    -------
    is_valid : bool
        True if the MDP structure is valid, False otherwise.
    error_msg : str
        Empty string if valid, otherwise an error message describing which
        state-action pairs have invalid probability distributions.
    """
    errors = []
    
    nS = mdp.num_states()
    nA = mdp.num_actions()
    
    # Check that transition probabilities form valid distributions
    # For each (s, a), probabilities over next states must sum to 1.0
    for s in range(nS):
        for a in range(nA):
            prob_sum = mdp.P[s, a, :].sum()
            if abs(prob_sum - 1.0) > tol and prob_sum > 0:
                errors.append(
                    f"P[{s}, {a}, :] sums to {prob_sum:.6f}, expected 1.0"
                )
    
    if errors:
        return False, "; ".join(errors[:5])  # Limit error messages for readability
    return True, ""


def check_policy_validity(
    mdp: InventoryMDP,
    policy: np.ndarray
) -> tuple[bool, str]:
    """
    Check if a policy is valid for the given MDP.
    
    This function validates that a policy array has the correct shape, contains
    integer actions, and that all actions are within the valid action space
    for the MDP. This is useful for catching errors before running algorithms
    that require valid policies.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance for which the policy should be valid.
    policy : np.ndarray
        Policy array of shape (n_states,). policy[s] should be the action to
        take in state s.
    
    Returns
    -------
    is_valid : bool
        True if the policy is valid for the MDP, False otherwise.
    error_msg : str
        Empty string if valid, otherwise an error message describing validation
        failures (shape, dtype, or invalid actions).
    """
    errors = []
    
    # Check policy shape matches the number of states
    if policy.shape != (mdp.num_states(),):
        errors.append(
            f"Policy shape {policy.shape} != expected ({mdp.num_states()},)"
        )
    
    # Check policy contains integer actions
    if policy.dtype != np.int_:
        errors.append("Policy must contain integer actions")
    
    # Check all actions are within valid range
    for s, a in enumerate(policy):
        if a < 0 or a >= mdp.num_actions():
            errors.append(
                f"Invalid action {a} in state {s} "
                f"(valid range: 0-{mdp.num_actions()-1})"
            )
    
    if errors:
        return False, "; ".join(errors[:5])
    return True, ""

