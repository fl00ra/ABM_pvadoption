import numpy as np
from collections import defaultdict

from agent_state import AgentArray
from network import compute_social_weights
from data_loader import load_amsterdam_data
from intervention import Policy


class ABM:
    """
    Agent-based model for PV adoption.

    Economic value V is computed in agent_logic.
    Adoption probability P is
        P_i = logistic(beta0_i + beta1 * V_norm_i + beta2 * S_i).
    """

    def _resolve_external_anchor(self, anchor_spec):
        """
        Resolve an external V anchor specification.
        """
        if anchor_spec is None:
            return None

        s = float(anchor_spec["s"])
        mu_star = float(anchor_spec["mu_star"])
        sd_star = float(anchor_spec["sd_star"])
        return (s, mu_star, sd_star)

    def __init__(
        self,
        n_agents,
        beta,
        behavior_mode,
        enable_feed_change=True,
        fit_mode="declining_net_metering",
        fit_params=None,
        load_data=load_amsterdam_data,
        adjacency_matrix=None,
        group_ids=None,
        min_r=0.03,
        max_r=0.14,
        precomputed_social_weight=None,
        record_history=True,
        store_distributions=True,
        sbi_collect=False,
        sbi_bins=5,
        external_V_anchor=None,
        seed=None,
        run_label=None,
        print_every=5,
    ):
        """
        Initialize the PV adoption ABM with agents, policy, and network.
        """
        self.n_agents = n_agents
        self.beta = beta

        self.policy = Policy(
            behavior_mode=behavior_mode,
            enable_feed_change=enable_feed_change,
            fit_mode=fit_mode,
            fit_params=fit_params,
        )

        self.seed = seed
        self.run_label = run_label
        self.print_every = int(print_every) if print_every is not None else 0

        self.agent_data = load_data(n_agents, policy=self.policy)
        self.n_agents = min(self.n_agents, len(self.agent_data))

        self.min_r = min_r
        self.max_r = max_r

        self.record_history = bool(record_history)
        self.store_distributions = bool(store_distributions)
        self.sbi_collect = bool(sbi_collect)
        self.sbi_bins = int(sbi_bins)

        self.external_V_anchor = external_V_anchor
        self._resolved_anchor = self._resolve_external_anchor(external_V_anchor)

        self.agents = AgentArray(self.n_agents, self.agent_data, self)
        if group_ids is not None:
            self.agents.group_id = group_ids

        if self._resolved_anchor is not None:
            s, mu_star, sd_star = self._resolved_anchor
            self.agents.set_external_V_anchor(s, mu_star, sd_star)

        self.adj_matrix = adjacency_matrix

        if precomputed_social_weight is not None:
            self.agents.social_weight = precomputed_social_weight
        else:
            self.agents.social_weight = compute_social_weights(self.adj_matrix)

        self.results = {
            "adoption_rate": [],
            "new_adopters": [],
            "adopters": [],
            "new_adoption_rate": [],
            "group_adoption": defaultdict(list),
            "distributions": {"V": [], "S": [], "P": []},
            "elec_price": [],
            "sbi": defaultdict(list),
            "group_adoption_ownership": defaultdict(list),
            "group_adoption_woning": defaultdict(list),
        }

        self.results["meta"] = {
            "seed": self.seed,
            "run_label": self.run_label,
            "beta": tuple(self.beta),
            "behavior_mode": self.policy.behavior_mode,
            "fit_mode": getattr(self.policy, "fit_mode", None),
            "enable_feed_change": bool(
                getattr(self.policy, "enable_feed_change", True)
            ),
            "min_r": float(self.min_r),
            "max_r": float(self.max_r),
            "external_V_anchor": (
                {
                    "s": self._resolved_anchor[0],
                    "mu_star": self._resolved_anchor[1],
                    "sd_star": self._resolved_anchor[2],
                }
                if self._resolved_anchor is not None
                else None
            ),
            "n_agents": int(self.n_agents),
        }

        self.agent_history = []
        self._prev_adopted_flags = self.agents.adopted.copy()

    def step(self, t: int) -> int:
        """
        Advance the simulation by one time step and update adoption.
        """
        if self.policy.behavior_mode == "tiered_pricing":
            elec_price_vec = self.policy.get_congestion_price_vectorized(
                elek_usage=self.agents.elek_usage,
                timestep=t,
            )
            self.agents.elec_price = elec_price_vec
            self.results["elec_price"].append(float(np.mean(elec_price_vec)))
        else:
            self.results["elec_price"].append(0.2)

        self.agents.update_V(t, self.policy, min_r=self.min_r, max_r=self.max_r)
        self.agents.update_S(self.adj_matrix)
        self.agents.update_P()

        new_adopters = self.agents.update_adoption(t)
        return new_adopters

    def run(self, n_steps: int, include_baseline: bool = True) -> None:
        """
        Run the simulation for a given number of steps.
        """
        if include_baseline:
            self._record_baseline()
            steps_to_run = max(0, n_steps - 1)
            start_t = 1
        else:
            steps_to_run = n_steps
            start_t = 0

        for k in range(steps_to_run):
            t = start_t + k
            self.step(t)
            self._record(t)

            if self.print_every and (t % self.print_every == 0):
                print(f"\n[Time {t}] Policy: {self.policy.behavior_mode}")
                print(f"Adoption rate: {self.results['adoption_rate'][-1]:.2%}")

    def _record_baseline(self) -> None:
        """
        Compute baseline (t=0), set V anchors if needed, and record initial stats.
        """
        self.agents.update_V(0, self.policy, min_r=self.min_r, max_r=self.max_r)

        if self._resolved_anchor is None:
            self.agents.set_V_anchor()

        self.agents.update_S(self.adj_matrix)
        self.agents.update_P()

        n_adopted = int(np.sum(self.agents.adopted))
        self.results["adoption_rate"].append(n_adopted / self.n_agents)
        self.results["new_adopters"].append(0)
        self.results["adopters"].append(n_adopted)
        self.results["new_adoption_rate"].append(0.0)

        for g in np.unique(self.agents.income_group):
            idxs = np.where(self.agents.income_group == g)[0]
            self.results["group_adoption"][g].append(
                float(np.mean(self.agents.adopted[idxs])),
            )

        for g in np.unique(self.agents.ownership_status):
            idxs = np.where(self.agents.ownership_status == g)[0]
            if idxs.size:
                self.results["group_adoption_ownership"][str(g)].append(
                    float(np.mean(self.agents.adopted[idxs])),
                )

        for g in np.unique(self.agents.woning_type):
            idxs = np.where(self.agents.woning_type == g)[0]
            if idxs.size:
                self.results["group_adoption_woning"][str(g)].append(
                    float(np.mean(self.agents.adopted[idxs])),
                )

        if self.store_distributions:
            self.results["distributions"]["V"].append(self.agents.V.copy())
            self.results["distributions"]["S"].append(self.agents.S.copy())
            self.results["distributions"]["P"].append(self.agents.P.copy())

        if self.record_history:
            agent_df = self.agents.to_dataframe()
            agent_df["timestep"] = 0
            agent_df["V"] = self.agents.V
            agent_df["S"] = self.agents.S
            agent_df["P"] = self.agents.P
            agent_df["beta0"] = self.agents.beta0
            agent_df["woning_type"] = self.agents.woning_type
            agent_df["woz_value"] = self.agents.woz_value
            agent_df["ownership_status"] = self.agents.ownership_status

            hh_rev = {v: k for k, v in self.agents.household_type_map.items()}
            agent_df["household_type"] = [
                hh_rev.get(idx, "") for idx in self.agents.household_type_idx
            ]
            agent_df["group_id"] = getattr(
                self.agents,
                "group_id",
                np.full(self.n_agents, -1, dtype=int),
            )

            b1 = float(self.beta[0])
            b2 = float(self.beta[1])

            s = float(self.agents.V_anchor_s)
            mu_star = float(self.agents.V_anchor_mu_star)
            sd_star = float(self.agents.V_anchor_sd_star)

            V_star = np.arcsinh(self.agents.V / s)
            V_norm = (V_star - mu_star) / (sd_star + 1e-12)

            agent_df["V_norm"] = V_norm
            agent_df["V_contrib"] = b1 * V_norm
            agent_df["S_contrib"] = b2 * self.agents.S
            agent_df["econ_term_raw"] = b1 * self.agents.V

            agent_df["is_new_adopter"] = False
            agent_df["seed"] = self.seed
            agent_df["run_label"] = self.run_label

            self.agent_history.append(agent_df)

        self._prev_adopted_flags = self.agents.adopted.copy()

    def _record(self, t: int) -> None:
        """
        Record statistics at timestep t (t >= 1).
        """
        n_adopted = int(np.sum(self.agents.adopted))

        prev_flags = self._prev_adopted_flags
        is_new = (self.agents.adopted) & (~prev_flags)
        new_adopters = int(is_new.sum())

        self.results["adoption_rate"].append(n_adopted / self.n_agents)
        self.results["new_adopters"].append(new_adopters)
        self.results["adopters"].append(n_adopted)
        self.results["new_adoption_rate"].append(new_adopters / self.n_agents)

        for g in np.unique(self.agents.income_group):
            idxs = np.where(self.agents.income_group == g)[0]
            self.results["group_adoption"][g].append(
                float(np.mean(self.agents.adopted[idxs])),
            )

        for g in np.unique(self.agents.ownership_status):
            idxs = np.where(self.agents.ownership_status == g)[0]
            if idxs.size:
                self.results["group_adoption_ownership"][str(g)].append(
                    float(np.mean(self.agents.adopted[idxs])),
                )

        for g in np.unique(self.agents.woning_type):
            idxs = np.where(self.agents.woning_type == g)[0]
            if idxs.size:
                self.results["group_adoption_woning"][str(g)].append(
                    float(np.mean(self.agents.adopted[idxs])),
                )

        if self.store_distributions:
            self.results["distributions"]["V"].append(self.agents.V.copy())
            self.results["distributions"]["S"].append(self.agents.S.copy())
            self.results["distributions"]["P"].append(self.agents.P.copy())

        if self.record_history:
            agent_df = self.agents.to_dataframe()
            agent_df["timestep"] = t
            agent_df["V"] = self.agents.V
            agent_df["S"] = self.agents.S
            agent_df["P"] = self.agents.P
            agent_df["beta0"] = self.agents.beta0
            agent_df["woning_type"] = self.agents.woning_type
            agent_df["woz_value"] = self.agents.woz_value
            agent_df["ownership_status"] = self.agents.ownership_status

            hh_rev = {v: k for k, v in self.agents.household_type_map.items()}
            agent_df["household_type"] = [
                hh_rev.get(idx, "") for idx in self.agents.household_type_idx
            ]
            agent_df["group_id"] = getattr(
                self.agents,
                "group_id",
                np.full(self.n_agents, -1, dtype=int),
            )

            b1 = float(self.beta[0])
            b2 = float(self.beta[1])

            s = float(self.agents.V_anchor_s)
            mu_star = float(self.agents.V_anchor_mu_star)
            sd_star = float(self.agents.V_anchor_sd_star)

            V_star = np.arcsinh(self.agents.V / s)
            V_norm = (V_star - mu_star) / (sd_star + 1e-12)

            agent_df["V_norm"] = V_norm
            agent_df["V_contrib"] = b1 * V_norm
            agent_df["S_contrib"] = b2 * self.agents.S
            agent_df["econ_term_raw"] = b1 * self.agents.V

            agent_df["is_new_adopter"] = is_new
            agent_df["seed"] = self.seed
            agent_df["run_label"] = self.run_label

            self.agent_history.append(agent_df)

        if self.sbi_collect:
            self._update_sbi_hazards(
                prev_flags,
                self.agents.adopted,
                self.agents.V,
                self.agents.S,
                step=t,
            )

        self._prev_adopted_flags = self.agents.adopted.copy()

        at_risk = int((~prev_flags).sum())
        cond_hazard = (
            self.results["new_adopters"][-1] / at_risk if at_risk > 0 else 0.0
        )
        self.results.setdefault("cond_hazard", []).append(cond_hazard)

    def _update_sbi_hazards(self, prev_flags, curr_flags, V, S, step: int) -> None:
        """
        Update per-quantile hazards for economic and social dimensions.
        """
        base_mask = ~prev_flags
        bins = max(2, int(self.sbi_bins))
        self.sbi_bins = bins

        if not np.any(base_mask):
            for b in range(self.sbi_bins):
                self.results["sbi"][f"haz_econ_bin{b}"].append(0.0)
                self.results["sbi"][f"haz_soc_bin{b}"].append(0.0)
            return

        V_base = V[base_mask]
        S_base = S[base_mask]
        new_base = (curr_flags & (~prev_flags))[base_mask]

        q_idx = np.linspace(0, 100, self.sbi_bins + 1)
        V_edges = np.nanpercentile(V_base, q_idx)
        S_edges = np.nanpercentile(S_base, q_idx)

        V_bin = np.clip(
            np.digitize(V_base, V_edges[1:-1], right=True),
            0,
            self.sbi_bins - 1,
        )
        S_bin = np.clip(
            np.digitize(S_base, S_edges[1:-1], right=True),
            0,
            self.sbi_bins - 1,
        )

        for b in range(self.sbi_bins):
            m = V_bin == b
            n = int(m.sum())
            k = int((new_base & m).sum())
            self.results["sbi"][f"haz_econ_bin{b}"].append(k / n if n > 0 else 0.0)

        for b in range(self.sbi_bins):
            m = S_bin == b
            n = int(m.sum())
            k = int((new_base & m).sum())
            self.results["sbi"][f"haz_soc_bin{b}"].append(k / n if n > 0 else 0.0)

    def get_results(self) -> dict:
        """
        Return the results dictionary collected during the run.
        """
        return self.results

    def get_timeseries(self):
        """
        Assemble a time-series DataFrame from results for aggregation across seeds.
        """
        import pandas as pd

        T = len(self.results["adoption_rate"])
        df = pd.DataFrame(
            {
                "timestep": list(range(T)),
                "adoption_rate": self.results["adoption_rate"],
                "new_adoption_rate": self.results["new_adoption_rate"],
                "adopters": self.results["adopters"],
                "new_adopters": self.results["new_adopters"],
                "elec_price": self.results["elec_price"][:T],
            }
        )
        meta = self.results.get("meta", {})
        for k, v in meta.items():
            df[k] = v
        return df
