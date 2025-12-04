# Healthcare Inventory MDP

A finite Markov Decision Process (MDP) model for hospital drug inventory
management, implemented in Python using only `numpy`.

The project follows the coursework structure for:

- Markov Processes
- Markov Decision Processes (MDPs)
- Value Iteration and Policy Iteration

## Structure

- `src/mdp_inventory.py` – InventoryMDP class with P and R matrices  
- `src/solvers.py` – value_iteration, policy_iteration, policy_evaluation  
- `src/baseline_policies.py` – (s,S) and other heuristic policies  
- `src/simulation.py` – simulation and metrics  
- `src/config.py` – default parameter configuration  
- `notebooks/experiments.ipynb` – experiments and plots  

## Running

```bash
python -m pip install numpy

Then in a Python session:

from src.config import default_params
from src.mdp_inventory import InventoryMDP
from src.solvers import value_iteration
from src.simulation import simulate_policy

params = default_params()
mdp = InventoryMDP(params)
V, policy, iters = value_iteration(mdp)
metrics = simulate_policy(mdp, policy, T=10000, initial_state=5, seed=0)
print(metrics)
```
