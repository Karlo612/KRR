from __future__ import annotations

"""
solvers.py

Dynamic programming algorithms for solving finite-state MDPs.

This module provides implementations of standard dynamic programming algorithms
for computing optimal policies in Markov Decision Processes. All algorithms
operate on InventoryMDP instances and utilise the precomputed transition
probability matrix P[s, a, s'] and expected reward matrix R_sa[s, a].

The module implements three core algorithms:
- Value Iteration: Iteratively updates the value function using the Bellman
  optimality equation until convergence to the optimal value function.
- Policy Evaluation: Computes the value function for a fixed policy by solving
  a linear system of equations.
- Policy Iteration: Alternates between policy evaluation and policy improvement
  steps until the policy converges to the optimal policy.

Mathematical Background:
    Value Iteration solves the Bellman optimality equation:
        V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
    
    Policy Iteration solves the policy evaluation equation:
        V^π(s) = R^π(s) + γ Σ_s' P^π(s'|s) V^π(s')
    followed by greedy policy improvement.
"""

from typing import Tuple
import warnings
import numpy as np
from .mdp_inventory import InventoryMDP


def value_iteration(
    mdp: InventoryMDP,
    tol: float = 1e-6,
    max_iter: int = 10_000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Solve the MDP using the value iteration algorithm.
    
    Value iteration computes the optimal value function by iteratively applying
    the Bellman optimality equation. At each iteration, the value function is
    updated by computing the maximum expected value over all actions:
    
        V_{k+1}(s) = max_a [R_sa(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]
    
    The algorithm converges when the change in value function falls below the
    tolerance threshold, measured using the L∞ norm: ||V_{k+1} - V_k||_∞ < tol.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance with precomputed transition and reward matrices.
    tol : float, optional
        Convergence tolerance for the value function. The algorithm stops when
        the maximum change in value function is less than tol. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations to prevent infinite loops. If convergence
        is not reached within max_iter iterations, a warning is issued.
        Default is 10,000.
    
    Returns
    -------
    V : np.ndarray
        Optimal value function of shape (n_states,). V[s] gives the optimal
        expected discounted reward starting from state s.
    policy : np.ndarray
        Optimal policy of shape (n_states,). policy[s] gives the optimal action
        to take in state s.
    iterations : int
        Number of iterations performed until convergence, or max_iter if
        convergence was not achieved.
    
    Notes
    -----
    The optimal policy is extracted by selecting the action that maximises the
    Q-value: π*(s) = argmax_a Q(s,a), where Q(s,a) = R_sa(s,a) + γ Σ_s' P(s'|s,a) V*(s').
    
    Computational Complexity:
        O(nS * nA * nS) per iteration, where nS = num_states, nA = num_actions.
        Typically converges in O(log(1/tol)) iterations for well-conditioned MDPs.
    """

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    
    # Validate matrix dimensions to ensure they match the MDP structure
    if mdp.P.shape != (nS, nA, nS):
        raise ValueError(
            f"P shape {mdp.P.shape} != expected ({nS}, {nA}, {nS})"
        )
    if mdp.R_sa.shape != (nS, nA):
        raise ValueError(
            f"R_sa shape {mdp.R_sa.shape} != expected ({nS}, {nA})"
        )
    
    P = mdp.P
    R_sa = mdp.R_sa

    # Initialise value function to zero (common starting point)
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    for it in range(max_iter):
        # Compute Q-values for all state-action pairs using vectorised operations
        # Q(s,a) = R_sa(s,a) + γ * Σ_s' P(s'|s,a) * V(s')
        # Using einsum for efficient matrix multiplication across state transitions
        Q = R_sa + γ * np.einsum('san,n->sa', P, V)
        
        # Extract optimal policy by selecting actions that maximise Q-values
        policy = np.argmax(Q, axis=1)
        
        # Update value function to the maximum Q-value for each state
        V_new = np.max(Q, axis=1)

        # Check convergence using L∞ norm (maximum change across all states)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, policy, it + 1

        V = V_new

    # Return result even if not converged (reached max_iter)
    final_residual = np.max(np.abs(V_new - V))
    warnings.warn(
        f"Value iteration did not converge after {max_iter} iterations. "
        f"Final residual: {final_residual:.2e} > tol={tol:.2e}. "
        f"Consider increasing max_iter or checking convergence criteria.",
        RuntimeWarning
    )
    return V, policy, max_iter


def policy_evaluation(mdp: InventoryMDP, policy: np.ndarray) -> np.ndarray:
    """
    Evaluate a fixed policy by computing its value function.
    
    For a fixed policy π, the value function V^π represents the expected
    discounted reward when following policy π from each state. It satisfies
    the Bellman equation for the given policy:
    
        V^π = R^π + γ P^π V^π
    
    Rearranging this equation yields a linear system:
        (I - γ P^π) V^π = R^π
    
    This linear system is solved directly to obtain the exact value function,
    which is more computationally expensive than one value iteration step but
    provides an exact solution.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance with precomputed transition and reward matrices.
    policy : np.ndarray
        Policy array of shape (n_states,). policy[s] specifies the action to
        take in state s.
    
    Returns
    -------
    V : np.ndarray
        Value function for the given policy, of shape (n_states,). V[s] gives
        the expected discounted reward when following the policy from state s.
    
    Raises
    ------
    ValueError
        If the policy is invalid (incorrect shape or actions outside valid range).
    
    Computational Complexity:
        O(nS^3) for solving the linear system, where nS = num_states.
        This is more expensive than one value iteration step but gives an exact
        solution rather than an approximation.
    """

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    
    # Validate matrix dimensions to ensure they match the MDP structure
    if mdp.P.shape != (nS, nA, nS):
        raise ValueError(
            f"P shape {mdp.P.shape} != expected ({nS}, {nA}, {nS})"
        )
    if mdp.R_sa.shape != (nS, nA):
        raise ValueError(
            f"R_sa shape {mdp.R_sa.shape} != expected ({nS}, {nA})"
        )
    
    P = mdp.P
    R_sa = mdp.R_sa

    # Validate that the policy has the correct shape
    if policy.shape != (nS,):
        raise ValueError(f"Policy shape {policy.shape} != ({nS},)")
    
    # Build policy-specific transition and reward matrices
    # For a fixed policy, we extract the relevant transitions and rewards
    P_pi = np.zeros((nS, nS))
    R_pi = np.zeros(nS)

    for s in range(nS):
        a = policy[s]
        if a < 0 or a >= mdp.num_actions():
            raise ValueError(f"Invalid action {a} in state {s}")
        
        # Extract transition probabilities for the policy's action in state s
        P_pi[s, :] = P[s, a, :]
        
        # Extract expected reward for the policy's action in state s
        R_pi[s] = R_sa[s, a]

    # Solve the linear system (I - γ P^π) V^π = R^π to get the value function
    A = np.eye(nS) - γ * P_pi
    b = R_pi

    return np.linalg.solve(A, b)


def policy_iteration(
    mdp: InventoryMDP,
    max_iter: int = 1_000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Solve the MDP using the policy iteration algorithm.
    
    Policy iteration finds the optimal policy by alternating between two steps:
    1. Policy Evaluation: Compute the value function V^π for the current policy π
    2. Policy Improvement: Update the policy greedily: π(s) = argmax_a Q(s,a)
    
    The algorithm continues until the policy stabilises (no changes occur),
    at which point the optimal policy has been found. Policy iteration typically
    converges in fewer iterations than value iteration, but each iteration is
    more computationally expensive as it requires solving a linear system.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance with precomputed transition and reward matrices.
    max_iter : int, optional
        Maximum number of policy iterations to prevent infinite loops. If the
        policy does not converge within max_iter iterations, a warning is issued.
        Default is 1,000.
    
    Returns
    -------
    V : np.ndarray
        Optimal value function of shape (n_states,). V[s] gives the optimal
        expected discounted reward starting from state s.
    policy : np.ndarray
        Optimal policy of shape (n_states,). policy[s] gives the optimal action
        to take in state s.
    iterations : int
        Number of policy iterations performed until convergence, or max_iter if
        convergence was not achieved.
    
    Notes
    -----
    Policy iteration typically converges in fewer iterations than value iteration,
    but each iteration is more expensive as it requires solving a linear system
    of equations during the policy evaluation step.
    
    Computational Complexity:
        O(nS^3) per policy evaluation step (linear system solve).
        O(nS * nA * nS) per policy improvement step.
        Typically converges in O(log(nS)) iterations.
    """

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    
    # Validate matrix dimensions to ensure they match the MDP structure
    if mdp.P.shape != (nS, nA, nS):
        raise ValueError(
            f"P shape {mdp.P.shape} != expected ({nS}, {nA}, {nS})"
        )
    if mdp.R_sa.shape != (nS, nA):
        raise ValueError(
            f"R_sa shape {mdp.R_sa.shape} != expected ({nS}, {nA})"
        )
    
    P = mdp.P
    R_sa = mdp.R_sa

    # Initialise policy (starting with zero policy; could also use random initialisation)
    policy = np.zeros(nS, dtype=int)

    for it in range(max_iter):
        # Policy Evaluation: compute the value function for the current policy
        V = policy_evaluation(mdp, policy)
        
        # Policy Improvement: compute Q-values and update policy greedily
        # Compute Q-values for all state-action pairs using vectorised operations
        # Q(s,a) = R_sa(s,a) + γ * Σ_s' P(s'|s,a) * V(s')
        Q = R_sa + γ * np.einsum('san,n->sa', P, V)
        
        # Update policy to select actions that maximise Q-values
        new_policy = np.argmax(Q, axis=1)
        
        # Check if policy has stabilised (converged to optimal policy)
        if np.array_equal(new_policy, policy):
            return V, new_policy, it + 1
        
        policy = new_policy

    # Return result even if not converged
    warnings.warn(
        f"Policy iteration did not converge after {max_iter} iterations. "
        f"Consider increasing max_iter.",
        RuntimeWarning
    )
    return V, policy, max_iter