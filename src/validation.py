"""
validation.py

Validation functions for MDP parameters and structure.
"""

from __future__ import annotations
import numpy as np
from .mdp_inventory import InventoryParams, InventoryMDP


def validate_params(params: InventoryParams) -> tuple[bool, str]:
    """
    Validate inventory MDP parameters.
    
    Parameters
    ----------
    params : InventoryParams
        Parameters to validate.
    
    Returns
    -------
    is_valid : bool
        True if all parameters are valid.
    error_msg : str
        Empty string if valid, otherwise error message.
    """
    errors = []
    
    # Check max_inventory
    if params.max_inventory < 1:
        errors.append("max_inventory must be >= 1")
    
    # Check max_order
    if params.max_order < 0:
        errors.append("max_order must be >= 0")
    
    # Check demand_probs
    if not params.demand_probs:
        errors.append("demand_probs cannot be empty")
    
    if any(d < 0 for d in params.demand_probs.keys()):
        errors.append("demand values must be non-negative")
    
    if any(p < 0 for p in params.demand_probs.values()):
        errors.append("demand probabilities must be non-negative")
    
    # Check costs
    if params.holding_cost < 0:
        errors.append("holding_cost must be non-negative")
    
    if params.shortage_cost < 0:
        errors.append("shortage_cost must be non-negative")
    
    if params.order_cost < 0:
        errors.append("order_cost must be non-negative")
    
    # Check discount factor
    if params.discount <= 0 or params.discount >= 1:
        errors.append("discount must be in (0, 1)")
    
    if errors:
        return False, "; ".join(errors)
    return True, ""


def validate_mdp(mdp: InventoryMDP, tol: float = 1e-6) -> tuple[bool, str]:
    """
    Validate MDP structure (transition probabilities sum to 1).
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP to validate.
    tol : float
        Tolerance for probability sum checks.
    
    Returns
    -------
    is_valid : bool
        True if MDP structure is valid.
    error_msg : str
        Empty string if valid, otherwise error message.
    """
    errors = []
    
    nS = mdp.num_states()
    nA = mdp.num_actions()
    
    # Check transition probabilities sum to 1
    for s in range(nS):
        for a in range(nA):
            prob_sum = mdp.P[s, a, :].sum()
            if abs(prob_sum - 1.0) > tol and prob_sum > 0:
                errors.append(
                    f"P[{s}, {a}, :] sums to {prob_sum:.6f}, expected 1.0"
                )
    
    if errors:
        return False, "; ".join(errors[:5])  # Limit error messages
    return True, ""


def check_policy_validity(
    mdp: InventoryMDP,
    policy: np.ndarray
) -> tuple[bool, str]:
    """
    Check if a policy is valid for the given MDP.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance.
    policy : np.ndarray
        Policy array, shape (n_states,).
    
    Returns
    -------
    is_valid : bool
        True if policy is valid.
    error_msg : str
        Empty string if valid, otherwise error message.
    """
    errors = []
    
    if policy.shape != (mdp.num_states(),):
        errors.append(
            f"Policy shape {policy.shape} != expected ({mdp.num_states()},)"
        )
    
    if policy.dtype != np.int_:
        errors.append("Policy must contain integer actions")
    
    for s, a in enumerate(policy):
        if a < 0 or a >= mdp.num_actions():
            errors.append(
                f"Invalid action {a} in state {s} "
                f"(valid range: 0-{mdp.num_actions()-1})"
            )
    
    if errors:
        return False, "; ".join(errors[:5])
    return True, ""

