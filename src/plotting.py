"""
plotting.py

Visualization functions for MDP results and comparisons.

This module provides functions to create publication-quality plots for:
- Policy comparisons
- Value functions
- Cost trajectories
- Inventory trajectories
- Performance metrics
"""

from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt


def plot_policy_comparison(
    policies: Dict[str, np.ndarray],
    states: Optional[np.ndarray] = None,
    title: str = "Policy Comparison",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot multiple policies for comparison.
    
    Parameters
    ----------
    policies : Dict[str, np.ndarray]
        Dictionary mapping policy names to policy arrays (shape: n_states,).
    states : np.ndarray, optional
        State values for x-axis. If None, uses indices.
    title : str, optional
        Plot title. Default is "Policy Comparison".
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if states is None:
        n_states = len(next(iter(policies.values())))
        states = np.arange(n_states)
    
    for name, policy in policies.items():
        ax.step(states, policy, where='mid', label=name, marker='o', markersize=4)
    
    ax.set_xlabel("State (Inventory Level)")
    ax.set_ylabel("Action (Order Quantity)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_value_function(
    V: np.ndarray,
    states: Optional[np.ndarray] = None,
    title: str = "Value Function",
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plot value function V(s).
    
    Parameters
    ----------
    V : np.ndarray
        Value function array, shape (n_states,).
    states : np.ndarray, optional
        State values for x-axis. If None, uses indices.
    title : str, optional
        Plot title. Default is "Value Function".
    figsize : tuple, optional
        Figure size. Default is (8, 5).
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if states is None:
        states = np.arange(len(V))
    
    ax.plot(states, V, marker='o', markersize=6, linewidth=2)
    ax.set_xlabel("State (Inventory Level)")
    ax.set_ylabel("V(s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_cost_comparison(
    metrics_dict: Dict[str, Dict],
    metric_name: str = "avg_cost_per_period",
    title: str = "Cost Comparison",
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plot bar chart comparing costs across policies.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict]
        Dictionary mapping policy names to metrics dictionaries.
    metric_name : str, optional
        Metric to plot. Default is "avg_cost_per_period".
    title : str, optional
        Plot title. Default is "Cost Comparison".
    figsize : tuple, optional
        Figure size. Default is (8, 5).
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    policies = list(metrics_dict.keys())
    values = [float(metrics_dict[p][metric_name]) for p in policies]
    
    x = np.arange(len(policies))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_cost_components(
    metrics_dict: Dict[str, Dict],
    title: str = "Cost Component Breakdown",
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot stacked or grouped bar chart of cost components.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict]
        Dictionary mapping policy names to metrics dictionaries.
    title : str, optional
        Plot title. Default is "Cost Component Breakdown".
    figsize : tuple, optional
        Figure size. Default is (10, 5).
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    policies = list(metrics_dict.keys())
    components = ['per_step_holding', 'per_step_shortage', 'per_step_ordering']
    component_labels = ['Holding', 'Shortage', 'Ordering']
    
    x = np.arange(len(policies))
    width = 0.35
    
    for i, (comp, label) in enumerate(zip(components, component_labels)):
        values = [
            float(metrics_dict[p][comp].sum() if hasattr(metrics_dict[p][comp], 'sum') 
                  else metrics_dict[p][comp]) 
            for p in policies
        ]
        offset = (i - 1) * width / len(components)
        ax.bar(x + offset, values, width / len(components), label=label)
    
    ax.set_xticks(x)
    ax.set_xticklabels(policies)
    ax.set_ylabel("Total Cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_trajectories(
    metrics_dict: Dict[str, Dict],
    trajectory_key: str = "inventory_levels",
    title: str = "Trajectories",
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot time series trajectories for multiple policies.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict]
        Dictionary mapping policy names to metrics dictionaries.
    trajectory_key : str, optional
        Key in metrics dict for trajectory data. Default is "inventory_levels".
    title : str, optional
        Plot title. Default is "Trajectories".
    figsize : tuple, optional
        Figure size. Default is (10, 5).
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, metrics in metrics_dict.items():
        traj = metrics[trajectory_key]
        if isinstance(traj, np.ndarray):
            ax.plot(traj, label=name, alpha=0.7)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel(trajectory_key.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_cumulative_cost(
    metrics_dict: Dict[str, Dict],
    title: str = "Cumulative Cost Over Time",
    figsize: tuple = (10, 5)
) -> plt.Figure:
    """
    Plot cumulative cost trajectories.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict]
        Dictionary mapping policy names to metrics dictionaries.
    title : str, optional
        Plot title. Default is "Cumulative Cost Over Time".
    figsize : tuple, optional
        Figure size. Default is (10, 5).
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, metrics in metrics_dict.items():
        if 'per_step_costs' in metrics:
            costs = metrics['per_step_costs']
            if isinstance(costs, np.ndarray):
                cum_costs = np.cumsum(costs)
                ax.plot(cum_costs, label=name)
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig

