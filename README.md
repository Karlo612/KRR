# Healthcare Inventory MDP

A finite Markov Decision Process (MDP) model for hospital drug inventory
management, implemented in Python using `numpy` and `matplotlib`.

The project follows the coursework structure for:

- Markov Processes
- Markov Decision Processes (MDPs)
- Value Iteration and Policy Iteration

## Structure

### Core Modules
- `src/mdp_inventory.py` – InventoryMDP class with P and R matrices  
- `src/solvers.py` – value_iteration, policy_iteration, policy_evaluation  
- `src/baseline_policies.py` – (s,S), fixed-order, and reorder-to-level policies  
- `src/simulation.py` – simulation and metrics collection  
- `src/config.py` – default parameter configuration  

### Utility Modules
- `src/validation.py` – parameter and MDP structure validation  
- `src/plotting.py` – visualization functions for results  
- `src/export.py` – export results to CSV/JSON formats  

### Experiments
- `notebooks/experiments.ipynb` – experiments and plots  

### Tests
- `tests/` – unit tests for core functionality

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.config import default_params
from src.mdp_inventory import InventoryMDP
from src.solvers import value_iteration
from src.simulation import simulate_policy

# Create MDP
params = default_params()
mdp = InventoryMDP(params)

# Solve using value iteration
V, policy, iters = value_iteration(mdp)
print(f"Converged in {iters} iterations")

# Simulate policy
def policy_fn(s): return int(policy[s])
metrics = simulate_policy(mdp, policy_fn, T=1000, initial_state=5, seed=0)
print(f"Average cost: {metrics['avg_cost_per_period']:.4f}")
print(f"Service level: {metrics['service_level']:.4f}")
```

## Features

- **MDP Implementation**: Full transition and reward matrix computation
- **Dynamic Programming**: Value iteration and policy iteration algorithms
- **Baseline Policies**: (s,S), fixed-order, and reorder-to-level heuristics
- **Simulation**: Comprehensive metrics collection and analysis
- **Validation**: Parameter and structure validation functions
- **Visualization**: Plotting functions for policies, value functions, and trajectories
- **Export**: CSV and JSON export for results
- **Testing**: Comprehensive unit test suite

## Running Tests

```bash
pytest tests/
```

## Example: Comparing Policies

```python
from src.baseline_policies import make_sS_policy
from src.plotting import plot_cost_comparison

# Optimal policy
V, policy_opt, _ = value_iteration(mdp)
def opt_policy(s): return int(policy_opt[s])

# Baseline policy
baseline = make_sS_policy(s=2, S=5)

# Simulate both
metrics_opt = simulate_policy(mdp, opt_policy, T=200, seed=42)
metrics_base = simulate_policy(mdp, baseline, T=200, seed=42)

# Compare
metrics_dict = {'Optimal': metrics_opt, 'Baseline': metrics_base}
plot_cost_comparison(metrics_dict)
```

## Documentation

All modules include comprehensive docstrings with:
- Mathematical formulations
- Parameter descriptions
- Usage examples
- Return value specifications
