# Agent-based model of residential solar PV adoption

This repository contains the core code for an ABM of residential solar PV adoption, developed as part of a Master's thesis in Computational Science (University of Amsterdam).

The model combines:
- Heterogeneous households (income, dwelling type, ownership, electricity use, property value)
- A homophily-controlled social network
- A economic evaluation of PV investments
- Policy regimes for subsidies, net-metering, and tiered electricity tariffs

The code here focuses on the model and simulation pipeline. It does not include any CBS data.

---

## Repository structure

The main modules are:

- `agent_logic.py`  
  Computes the net present value (NPV) of PV adoption for each agent, given system size, yield, consumption, and policy.

- `agent_state.py`  
  Defines the `AgentArray` container for all agent-level attributes and state updates:
  - sampling PV system sizes, yields, and prices
  - computing economic value `V`
  - computing social influence `S`
  - computing adoption probabilities `P`
  - updating adoption decisions over time

- `data_loader.py`  
  Loads or synthesizes household data based on aggregate tables.

- `intervention.py`  
  Implements the `Policy` class:
  - net-metering regimes (`no_net_metering`, `fixed_net_metering`, `declining_net_metering`)
  - investment subsidies (universal and income-targeted)
  - tiered retail tariffs and congestion pricing

- `network.py`  
  Builds multiple homophilic social networks over agents:
  - grouping by dwelling type and WOZ high/low
  - controlling intra-group link fractions (`intra_ratio`)
  - computing social weights for influence

- `model.py`  
  Implements the `ABM` class:
  - holds the agent population, policy, and adjacency matrix
  - runs the annual adoption loop
  - records aggregate time series and (optionally) full agent-level histories
  - supports external anchors for normalising the economic term

- `simulation.py` 
  Run large batches of simulations:
  - multiple seeds
  - multiple homophily levels
  - multiple policy scenarios
  - writes per-timestep agent histories to partitioned Parquet files and creates a manifest

- `config.py`  
  Defines global configuration parameters, such as:
  - `n_agents`: number of agents
  - `n_steps`: number of simulated years
  - `beta`: behavioural parameters for the economic and social terms

---

## Installation

Tested with Python 3.11.

Install dependencies:

```bash
pip install -r requirements.txt
