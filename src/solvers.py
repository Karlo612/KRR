from __future__ import annotations

"""
solvers.py

Dynamic programming algorithms for finite MDPs:

- Value Iteration: Iteratively updates value function until convergence
- Policy Evaluation: Solves linear system for given policy's value function
- Policy Iteration: Alternates between policy evaluation and improvement

All algorithms operate on an InventoryMDP instance and its
precomputed P[s, a, s'] and R_sa[s, a] matrices.

Mathematical Background:
    Value Iteration solves: V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]
    Policy Iteration solves: V^π(s) = R^π(s) + γ Σ_s' P^π(s'|s) V^π(s')
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
    Solve MDP using value iteration algorithm.
    
    Value iteration iteratively updates the value function using the Bellman
    optimality equation until convergence:
    
        V_{k+1}(s) = max_a [R_sa(s,a) + γ Σ_s' P(s'|s,a) V_k(s')]
    
    The algorithm terminates when ||V_{k+1} - V_k||_∞ < tol.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance with precomputed P and R_sa matrices.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 10,000.
    
    Returns
    -------
    V : np.ndarray
        Optimal value function, shape (n_states,).
    policy : np.ndarray
        Optimal policy, shape (n_states,). policy[s] = optimal action in state s.
    iterations : int
        Number of iterations until convergence (or max_iter if not converged).
    
    Notes
    -----
    The policy is extracted as: π*(s) = argmax_a Q(s,a)
    where Q(s,a) = R_sa(s,a) + γ Σ_s' P(s'|s,a) V*(s')
    """

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    
    # Shape validation guards
    assert mdp.P.shape == (nS, nA, nS), \
        f"P shape {mdp.P.shape} != expected ({nS}, {nA}, {nS})"
    assert mdp.R_sa.shape == (nS, nA), \
        f"R_sa shape {mdp.R_sa.shape} != expected ({nS}, {nA})"
    
    P = mdp.P
    R_sa = mdp.R_sa

    # Initialize value function
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    for it in range(max_iter):
        V_new = np.zeros_like(V)

        # Bellman update for each state
        for s in range(nS):
            # Compute Q-values for all actions
            Q_sa = np.zeros(nA)
            for a in range(nA):
                # Q(s,a) = R_sa(s,a) + γ Σ_s' P(s'|s,a) V(s')
                Q_sa[a] = R_sa[s, a] + γ * np.sum(P[s, a, :] * V)
            
            # Greedy action selection
            best_a = int(np.argmax(Q_sa))
            V_new[s] = Q_sa[best_a]
            policy[s] = best_a

        # Check convergence (L∞ norm)
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
    Evaluate a policy by solving the linear system for its value function.
    
    For a fixed policy π, the value function V^π satisfies:
    
        V^π = R^π + γ P^π V^π
    
    Rearranging: (I - γ P^π) V^π = R^π
    
    This is solved as a linear system.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance with precomputed P and R_sa matrices.
    policy : np.ndarray
        Policy array, shape (n_states,). policy[s] = action in state s.
    
    Returns
    -------
    V : np.ndarray
        Value function for the given policy, shape (n_states,).
    
    Raises
    ------
    ValueError
        If policy is invalid (wrong shape or invalid actions).
    """

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    
    # Shape validation guards
    assert mdp.P.shape == (nS, nA, nS), \
        f"P shape {mdp.P.shape} != expected ({nS}, {nA}, {nS})"
    assert mdp.R_sa.shape == (nS, nA), \
        f"R_sa shape {mdp.R_sa.shape} != expected ({nS}, {nA})"
    
    P = mdp.P
    R_sa = mdp.R_sa

    # Validate policy
    if policy.shape != (nS,):
        raise ValueError(f"Policy shape {policy.shape} != ({nS},)")
    
    # Build policy-specific transition and reward matrices
    P_pi = np.zeros((nS, nS))
    R_pi = np.zeros(nS)

    for s in range(nS):
        a = policy[s]
        if a < 0 or a >= mdp.num_actions():
            raise ValueError(f"Invalid action {a} in state {s}")
        
        # P^π(s, s') = P(s' | s, π(s))
        P_pi[s, :] = P[s, a, :]
        
        # R^π(s) = R_sa(s, π(s))
        R_pi[s] = R_sa[s, a]

    # Solve (I - γ P^π) V^π = R^π
    A = np.eye(nS) - γ * P_pi
    b = R_pi

    return np.linalg.solve(A, b)


def policy_iteration(
    mdp: InventoryMDP,
    max_iter: int = 1_000
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Solve MDP using policy iteration algorithm.
    
    Policy iteration alternates between:
    1. Policy Evaluation: Compute V^π for current policy π
    2. Policy Improvement: Update π(s) = argmax_a Q(s,a)
    
    The algorithm terminates when the policy no longer changes.
    
    Parameters
    ----------
    mdp : InventoryMDP
        MDP instance with precomputed P and R_sa matrices.
    max_iter : int, optional
        Maximum number of policy iterations. Default is 1,000.
    
    Returns
    -------
    V : np.ndarray
        Optimal value function, shape (n_states,).
    policy : np.ndarray
        Optimal policy, shape (n_states,). policy[s] = optimal action in state s.
    iterations : int
        Number of iterations until convergence (or max_iter if not converged).
    
    Notes
    -----
    Policy iteration typically converges in fewer iterations than value iteration,
    but each iteration is more expensive (requires solving a linear system).
    """

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    
    # Shape validation guards
    assert mdp.P.shape == (nS, nA, nS), \
        f"P shape {mdp.P.shape} != expected ({nS}, {nA}, {nS})"
    assert mdp.R_sa.shape == (nS, nA), \
        f"R_sa shape {mdp.R_sa.shape} != expected ({nS}, {nA})"
    
    P = mdp.P
    R_sa = mdp.R_sa

    # Initialize with zero policy (or could use random)
    policy = np.zeros(nS, dtype=int)

    for it in range(max_iter):
        # Policy Evaluation: solve for V^π
        V = policy_evaluation(mdp, policy)
        
        # Policy Improvement: update policy greedily
        stable = True
        for s in range(nS):
            old_a = policy[s]

            # Compute Q-values for all actions
            Q_sa = np.zeros(nA)
            for a in range(nA):
                # Q(s,a) = R_sa(s,a) + γ Σ_s' P(s'|s,a) V^π(s')
                Q_sa[a] = R_sa[s, a] + γ * np.sum(P[s, a, :] * V)

            # Greedy improvement
            new_a = int(np.argmax(Q_sa))
            policy[s] = new_a

            if new_a != old_a:
                stable = False

        # Check if policy is stable (converged)
        if stable:
            return V, policy, it + 1

    # Return result even if not converged
    warnings.warn(
        f"Policy iteration did not converge after {max_iter} iterations. "
        f"Consider increasing max_iter.",
        RuntimeWarning
    )
    return V, policy, max_iter