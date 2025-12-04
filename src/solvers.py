from __future__ import annotations

"""
solvers.py

Dynamic programming algorithms for finite MDPs:

- Value Iteration
- Policy Evaluation
- Policy Iteration

All algorithms operate on an InventoryMDP instance and its
P[s, a, s'] and R[s, a, s'] matrices.
"""

from typing import Tuple
import numpy as np
from .mdp_inventory import InventoryMDP


def value_iteration(
    mdp: InventoryMDP,
    tol: float = 1e-6,
    max_iter: int = 10_000
) -> Tuple[np.ndarray, np.ndarray, int]:

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    P = mdp.P
    R = mdp.R

    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    for it in range(max_iter):
        V_new = np.zeros_like(V)

        for s in range(nS):
            Q_sa = np.zeros(nA)
            for a in range(nA):
                Q_sa[a] = np.sum(P[s, a, :] * (R[s, a, :] + γ * V))
            best_a = int(np.argmax(Q_sa))
            V_new[s] = Q_sa[best_a]
            policy[s] = best_a

        if np.max(np.abs(V_new - V)) < tol:
            return V_new, policy, it + 1

        V = V_new

    return V, policy, max_iter


def policy_evaluation(mdp: InventoryMDP, policy: np.ndarray) -> np.ndarray:

    nS = mdp.num_states()
    γ = mdp.discount
    P = mdp.P
    R = mdp.R

    P_pi = np.zeros((nS, nS))
    R_pi = np.zeros(nS)

    for s in range(nS):
        a = policy[s]
        P_pi[s, :] = P[s, a, :]
        R_pi[s] = np.sum(P[s, a, :] * R[s, a, :])

    A = np.eye(nS) - γ * P_pi
    b = R_pi

    return np.linalg.solve(A, b)


def policy_iteration(
    mdp: InventoryMDP,
    max_iter: int = 1_000
) -> Tuple[np.ndarray, np.ndarray, int]:

    nS = mdp.num_states()
    nA = mdp.num_actions()
    γ = mdp.discount
    P = mdp.P
    R = mdp.R

    policy = np.zeros(nS, dtype=int)

    for it in range(max_iter):
        V = policy_evaluation(mdp, policy)
        stable = True

        for s in range(nS):
            old_a = policy[s]

            Q_sa = np.zeros(nA)
            for a in range(nA):
                Q_sa[a] = np.sum(P[s, a, :] * (R[s, a, :] + γ * V))

            new_a = int(np.argmax(Q_sa))
            policy[s] = new_a

            if new_a != old_a:
                stable = False

        if stable:
            return V, policy, it + 1

    return V, policy, max_iter