import numpy as np

ELEC_PRICE = ...         # €/kWh retail price (constant)
EXPORT_PRICE_RATIO = 0.3   # fraction of retail price for low feed-in


def compute_V(agent_array, timestep: int, policy, T: int = 25,
              min_r: float = 0.03, max_r: float = 0.14) -> np.ndarray:
    """
    Compute the net present value (NPV) of a PV investment for each agent.

    Parameters:
        agent_array : AgentArray
            Container with agent-level attributes (system size, income, demand, etc.).
        timestep : int
            Decision time (current year index in the simulation).
        policy : object
            Policy object providing pricing and cost functions.
        T : int, optional
            Investment horizon in years. Default is 25.
        min_r : float, optional
            Minimum discount rate for the richest households.
        max_r : float, optional
            Maximum discount rate for the poorest households.

    Returns:
        np.ndarray
            NPV of PV adoption for each agent (benefits minus effective cost).
    """
    n = agent_array.n_agents

    # Annual generation and consumption
    E_gen = agent_array.system_size * agent_array.y_pv
    E_cons = np.maximum(agent_array.elek_usage, 1e-6)

    # Simple self-consumption rate: fixed fraction of annual generation
    SCR = 0.3

    # Income-based discount rate (poorer households → higher r)
    income = np.clip(agent_array.income, 100, 100_000)
    norm_income = (income - 100) / (100_000 - 100)
    r = max_r - norm_income * (max_r - min_r)

    # Energy flow decomposition that does not depend on policy
    E_use_raw = SCR * E_gen
    E_use = np.minimum(E_use_raw, E_cons)                # on-site use
    E_export = np.maximum(E_gen - E_use, 0.0)            # exported to grid
    L_grid = np.maximum(E_cons - E_use, 0.0)             # residual demand from grid
    E_salder_raw = np.minimum(E_export, L_grid)          # potential for net-metering

    npv = np.zeros(n, dtype=float)

    for t in range(T):
        future_t = timestep + t

        nm_share = policy.get_net_metering_share_expectation(
            decision_t=timestep,
            future_t=future_t,
        )

        phi = np.asarray(nm_share, dtype=float)
        if phi.shape == ():
            phi = np.full(n, float(phi), dtype=float)

        # Split exports into net-metered part and low-price feed-in part
        E_salder = phi * E_salder_raw
        E_feed = E_export - E_salder

        if getattr(policy, "behavior_mode", "") == "tiered_pricing":
            # Tiered retail price + net-metering + low-price feed-in

            # Annual bill without PV
            bill0, _ = policy.block_bill_and_avg_price_fixed(E_cons)

            # Effective grid demand with PV and net-metering
            E_cons_pv = np.maximum(E_cons - E_use - E_salder, 0.0)
            bill1, _ = policy.block_bill_and_avg_price_fixed(E_cons_pv)

            savings_bill = bill0 - bill1

            feed_price_low = ELEC_PRICE * EXPORT_PRICE_RATIO
            annual_benefit = savings_bill + E_feed * feed_price_low

        else:
            elec_price = np.full_like(E_cons, ELEC_PRICE, dtype=float)
            feed_price_low = elec_price * EXPORT_PRICE_RATIO

            annual_benefit = (E_use + E_salder) * elec_price + E_feed * feed_price_low

        npv += annual_benefit / ((1.0 + r) ** (t + 1))

    # Up-front investment cost after policy support
    base_cost = agent_array.base_cost
    income_group = getattr(agent_array, "income_group", None)

    effective_costs = policy.get_effective_cost(
        base_cost,
        timestep,
        income_group=income_group,
    )

    return npv - effective_costs
