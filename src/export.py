"""
export.py

Functions to export MDP results to various formats (CSV, JSON).

This module provides utilities for saving experimental results, policies,
and metrics for further analysis or reporting.
"""

from __future__ import annotations
from typing import Dict, Any
import json
import csv
import numpy as np
from pathlib import Path


def export_metrics_to_csv(
    metrics_dict: Dict[str, Dict[str, Any]],
    filename: str,
    include_trajectories: bool = False
) -> None:
    """
    Export metrics to CSV file.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, Any]]
        Dictionary mapping policy names to metrics dictionaries.
    filename : str
        Output CSV filename.
    include_trajectories : bool, optional
        If True, includes per-step trajectories. Default is False.
    
    Examples
    --------
    >>> metrics = {
    ...     'optimal': {'total_cost': 100.0, 'avg_cost_per_period': 0.5},
    ...     'baseline': {'total_cost': 120.0, 'avg_cost_per_period': 0.6}
    ... }
    >>> export_metrics_to_csv(metrics, 'results.csv')
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Aggregate metrics (non-array values)
    rows = []
    for policy_name, metrics in metrics_dict.items():
        row = {'policy': policy_name}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                row[key] = value
            elif isinstance(value, np.ndarray) and not include_trajectories:
                # Skip arrays unless explicitly requested
                continue
            elif isinstance(value, np.ndarray) and include_trajectories:
                # Store array statistics
                row[f'{key}_mean'] = float(np.mean(value))
                row[f'{key}_std'] = float(np.std(value))
                row[f'{key}_min'] = float(np.min(value))
                row[f'{key}_max'] = float(np.max(value))
        rows.append(row)
    
    if rows:
        fieldnames = rows[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def export_policy_to_csv(
    policy: np.ndarray,
    filename: str,
    states: np.ndarray | None = None
) -> None:
    """
    Export policy to CSV file.
    
    Parameters
    ----------
    policy : np.ndarray
        Policy array, shape (n_states,).
    filename : str
        Output CSV filename.
    states : np.ndarray, optional
        State values. If None, uses indices.
    
    Examples
    --------
    >>> policy = np.array([4, 3, 2, 0, 0, 0])
    >>> export_policy_to_csv(policy, 'policy.csv')
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if states is None:
        states = np.arange(len(policy))
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['state', 'action'])
        for s, a in zip(states, policy):
            writer.writerow([int(s), int(a)])


def export_results_to_json(
    results: Dict[str, Any],
    filename: str
) -> None:
    """
    Export results dictionary to JSON file.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary. NumPy arrays will be converted to lists.
    filename : str
        Output JSON filename.
    
    Examples
    --------
    >>> results = {
    ...     'policy': [4, 3, 2, 0, 0, 0],
    ...     'value_function': [-10.5, -9.2, -8.1, -7.5, -7.0, -6.8],
    ...     'iterations': 239
    ... }
    >>> export_results_to_json(results, 'results.json')
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert_numpy(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj
    
    json_data = convert_numpy(results)
    
    with open(path, 'w') as f:
        json.dump(json_data, f, indent=2)


def export_metrics_summary(
    metrics_dict: Dict[str, Dict[str, Any]],
    filename: str
) -> None:
    """
    Export a summary table of metrics to CSV.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Dict[str, Any]]
        Dictionary mapping policy names to metrics dictionaries.
    filename : str
        Output CSV filename.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all metric keys
    all_keys = set()
    for metrics in metrics_dict.values():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                all_keys.add(key)
    
    # Build rows
    rows = []
    for policy_name, metrics in metrics_dict.items():
        row = {'policy': policy_name}
        for key in sorted(all_keys):
            if key in metrics and isinstance(metrics[key], (int, float)):
                row[key] = metrics[key]
        rows.append(row)
    
    if rows:
        fieldnames = ['policy'] + sorted([k for k in all_keys if k != 'policy'])
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

