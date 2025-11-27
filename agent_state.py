import numpy as np
import pandas as pd
from typing import Dict, Optional

from agent_logic import compute_V


class AgentArray:
    """
    Container for all agent-level attributes.
    """

    def __init__(self, n_agents: int, agent_data: list, model) -> None:
        """
        Initialize agent-level attributes.

        Parameters:
            n_agents: Number of agents.
            agent_data: List of dicts with agent attributes.
            model: Model instance providing RNG and beta.
        """
        self.n_agents = n_agents
        self.model = model
        self.agent_data = agent_data

        if hasattr(model, "rng") and model.rng is not None:
            self.rng = model.rng
        else:
            seed = getattr(model, "rng_seed", None)
            self.rng = np.random.default_rng(seed)

        self._initialize_arrays()
        self._load_from_data(agent_data)
        self._initialize_sampled_attributes()
        self._compute_derived_attributes()

    def _initialize_arrays(self) -> None:
        """Allocate arrays for core attributes."""
        n = self.n_agents

        self.ids = np.arange(n, dtype=np.int32)

        self.income = np.zeros(n, dtype=np.float32)
        self.income_level = np.zeros(n, dtype=np.int8)  # 0=low, 1=mid, 2=high
        self.income_group = np.zeros(n, dtype=np.int8)

        self.household_type_idx = np.zeros(n, dtype=np.int8)
        self.elek_usage = np.zeros(n, dtype=np.float32)

        self.system_size = np.zeros(n, dtype=np.float32)
        self.pv_price = np.zeros(n, dtype=np.float32)
        self.y_pv = np.zeros(n, dtype=np.float32)
        self.base_cost = np.zeros(n, dtype=np.float32)

        self.beta0 = np.array(
            [a["beta0"] for a in self.agent_data[:n]],
            dtype=np.float32,
        )
        self.social_weight = np.ones(n, dtype=np.float32)

        self.adopted = np.array(
            [a.get("adopted", False) for a in self.agent_data[:n]],
            dtype=bool,
        )
        self.adoption_time = np.full(n, -1, dtype=np.int16)
        self.is_targeted = np.zeros(n, dtype=bool)

        self.V = np.zeros(n, dtype=np.float32)
        self.S = np.zeros(n, dtype=np.float32)
        self.P = np.zeros(n, dtype=np.float32)

        self.V_mu0: Optional[float] = None
        self.V_sd0: Optional[float] = None

        self.household_type_map: Dict[str, int] = {}

        self.woning_type = np.empty(n, dtype=object)
        self.ownership_status = np.empty(n, dtype=object)
        self.woz_value = np.zeros(n, dtype=np.float32)

        self.V_anchor_s: Optional[float] = None
        self.V_anchor_mu_star: Optional[float] = None
        self.V_anchor_sd_star: Optional[float] = None

    def _load_from_data(self, agent_data: list) -> None:
        """Load basic attributes from the input dictionaries."""
        household_types = [d.get("household_type", "single") for d in agent_data]
        unique_household_types = sorted(set(household_types))
        self.household_type_map = {
            ht: idx for idx, ht in enumerate(unique_household_types)
        }

        for i, d in enumerate(agent_data[:self.n_agents]):
            self.income[i] = d.get("income", 30_000.0)
            self.household_type_idx[i] = self.household_type_map.get(
                d.get("household_type", "single"),
                0,
            )
            self.elek_usage[i] = d.get("elek_usage", 3_500.0)
            self.adopted[i] = d.get("adopted", False)
            self.ownership_status[i] = d.get("ownership_status", "private owner")
            self.woning_type[i] = d.get("woning_type", "unknown")
            self.woz_value[i] = d.get("woz_value", 300_000.0)

        self._categorize_income()

    def _categorize_income(self) -> None:
        """Assign income groups using empirical tertiles (0=low,1=mid,2=high)."""
        low_thres = np.percentile(self.income, 33.3)
        high_thres = np.percentile(self.income, 66.7)

        self.income_level[self.income < low_thres] = 0
        self.income_level[
            (self.income >= low_thres) & (self.income <= high_thres)
        ] = 1
        self.income_level[self.income > high_thres] = 2

        self.income_group = self.income_level.copy()

    def _initialize_sampled_attributes(self) -> None:
        """Sample PV-related technical attributes."""
        self._sample_ypv()
        self._sample_system_sizes()
        self._sample_pv_prices()

    def _sample_system_sizes(self) -> None:
        """Sample system sizes targeting annual coverage close to usage."""
        n = self.n_agents

        E_cons = np.maximum(self.elek_usage.astype(np.float64), 1e-6)
        coverage = self.rng.uniform(0.85, 1.15, size=n).astype(np.float64)
        E_target = E_cons * coverage

        y = np.maximum(self.y_pv.astype(np.float64), 1e-6)
        system_size = E_target / y
        system_size = np.clip(system_size, 0.5, 10.0)

        self.system_size = system_size.astype(np.float32)

    def _sample_pv_prices(self) -> None:
        """Sample PV prices (€/kWp) from a triangular distribution."""
        self.pv_price = self.rng.triangular(
            left=1100.0,
            mode=1300.0,
            right=1500.0,
            size=self.n_agents,
        ).astype(np.float32)

    def _sample_ypv(self) -> None:
        """Sample yearly PV yield (kWh/kWp/yr) from a triangular distribution."""
        self.y_pv = self.rng.triangular(
            left=850.0,
            mode=900.0,
            right=950.0,
            size=self.n_agents,
        ).astype(np.float32)

    def _compute_derived_attributes(self) -> None:
        """Compute base investment cost."""
        self.base_cost = self.system_size * self.pv_price

    def reseed(self, seed: int) -> None:
        """Reset the RNG with a new seed."""
        self.rng = np.random.default_rng(seed)

    def update_V(
        self,
        timestep: int,
        policy,
        min_r: Optional[float] = None,
        max_r: Optional[float] = None,
    ) -> None:
        """
        Update NPV of adoption for each agent.

        Parameters:
            timestep: Current simulation time.
            policy: Policy object with pricing and cost functions.
            min_r: Optional lower bound for discount rate.
            max_r: Optional upper bound for discount rate.
        """
        if min_r is None or max_r is None:
            self.V = compute_V(self, timestep, policy, T=20)
        else:
            self.V = compute_V(
                self,
                timestep,
                policy,
                T=20,
                min_r=min_r,
                max_r=max_r,
            )

    def set_V_anchor(self, fixed_s: Optional[float] = None) -> None:
        """
        Define asinh-based anchors from the current V distribution.

        If fixed_s is provided (in euros), use it as the scale s.
        Otherwise use a robust measure (MAD or std) as s.
        """
        V = np.asarray(self.V, dtype=float)

        if fixed_s is not None and fixed_s > 0:
            s = float(fixed_s)
        else:
            med = np.median(V)
            mad = np.median(np.abs(V - med))
            if mad > 0:
                s = float(mad)
            else:
                std = np.std(V)
                s = float(std if std > 0 else 1.0)

        V_star = np.arcsinh(V / s)
        mu_star = float(np.mean(V_star))
        sd_star = float(np.std(V_star)) if np.std(V_star) > 0 else 1.0

        self.V_anchor_s = s
        self.V_anchor_mu_star = mu_star
        self.V_anchor_sd_star = sd_star

        self.V_mu0 = mu_star
        self.V_sd0 = sd_star

    def set_external_V_anchor(self, s: float, mu_star: float, sd_star: float) -> None:
        """
        Use externally provided asinh-based anchors for V normalization.

        Parameters:
            s: Scale in euros for asinh(V / s).
            mu_star: Mean of V* on the reference distribution.
            sd_star: Standard deviation of V* on the reference distribution.
        """
        self.V_anchor_s = float(max(s, 1e-12))
        self.V_anchor_mu_star = float(mu_star)
        self.V_anchor_sd_star = float(max(sd_star, 1e-12))

        self.V_mu0 = self.V_anchor_mu_star
        self.V_sd0 = self.V_anchor_sd_star

    def update_S(self, adjacency_matrix) -> None:
        """
        Update social influence S based on neighbors' adoption.

        S_i is a weighted average of adopted neighbors.
        """
        weighted_adopted = self.adopted * self.social_weight

        weighted_sum = adjacency_matrix.dot(weighted_adopted)
        total_weight = adjacency_matrix.dot(self.social_weight)

        S = np.divide(
            weighted_sum,
            total_weight,
            out=np.zeros_like(weighted_sum),
            where=total_weight > 0,
        )

        self.S = S

    def update_P(self) -> None:
        """
        Update adoption probability P using asinh-based normalization of V.

        Uses beta0 (heterogeneous intercepts), and model.beta for V and S.
        """
        beta = self.model.beta

        if (
            self.V_anchor_s is None
            or self.V_anchor_mu_star is None
            or self.V_anchor_sd_star is None
        ):
            self.set_V_anchor()

        s = self.V_anchor_s
        mu_star = self.V_anchor_mu_star
        sd_star = self.V_anchor_sd_star

        V_star = np.arcsinh(self.V / s)
        V_norm = (V_star - mu_star) / (sd_star + 1e-12)

        S_norm = self.S

        logit = self.beta0 + beta[0] * V_norm + beta[1] * S_norm
        self.P = 1.0 / (1.0 + np.exp(-logit))

    def update_adoption(self, timestep: int) -> int:
        """
        Draw adoption events based on P and tenant eligibility.

        Returns:
            Number of new adopters at this timestep.
        """
        rand_vals = self.rng.random(self.n_agents)

        behavior_mode = getattr(self.model.policy, "behavior_mode", "default")
        allow_tenants = behavior_mode == "open_to_tenants"

        if allow_tenants:
            new_adopt = (~self.adopted) & (rand_vals < self.P)
        else:
            is_owner = self.ownership_status == "private owner"
            new_adopt = (~self.adopted) & is_owner & (rand_vals < self.P)

        self.adopted[new_adopt] = True
        self.adoption_time[new_adopt] = timestep

        return int(np.sum(new_adopt))

    def get_feature_matrix(self) -> np.ndarray:
        """
        Build a feature matrix for similarity-based network construction.

        Returns:
            Feature matrix of shape (n_agents, n_features).
        """
        OWNERSHIP_VOCAB = [
            "private owner",
            "housing corporation",
            "institutional investor",
            "unknown",
        ]
        WONING_VOCAB = [
            "apartment",
            "detached",
            "semi-detached",
            "mid-terrace",
            "corner house",
            "unknown",
        ]

        own_to_idx = {c: i for i, c in enumerate(OWNERSHIP_VOCAB)}
        woning_to_idx = {c: i for i, c in enumerate(WONING_VOCAB)}

        n = self.n_agents
        own_onehot = np.zeros((n, len(OWNERSHIP_VOCAB)), dtype=np.float32)
        woning_onehot = np.zeros((n, len(WONING_VOCAB)), dtype=np.float32)

        for i in range(n):
            own = self.ownership_status[i] or "unknown"
            own_col = own_to_idx.get(own, own_to_idx["unknown"])
            own_onehot[i, own_col] = 1.0

            wt = self.woning_type[i] or "unknown"
            wt_col = woning_to_idx.get(wt, woning_to_idx["unknown"])
            woning_onehot[i, wt_col] = 1.0

        hh_idx = self.household_type_idx.astype(np.float32).reshape(-1, 1)
        income_lvl = self.income_level.astype(np.float32).reshape(-1, 1)
        elek_scaled = (self.elek_usage / 1000.0).astype(np.float32).reshape(-1, 1)

        X = np.concatenate(
            [hh_idx, income_lvl, elek_scaled, own_onehot, woning_onehot],
            axis=1,
        )

        self.feature_vocabs = {
            "ownership_vocab": OWNERSHIP_VOCAB,
            "woning_vocab": WONING_VOCAB,
            "household_note": (
                "household_type_idx is an internal mapping built from data at load time."
            ),
        }

        return X

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export core agent attributes to a pandas DataFrame.
        """
        household_type_reverse = {v: k for k, v in self.household_type_map.items()}
        income_str_map = np.array(["low", "mid", "high"])

        data = {
            "id": self.ids,
            "income": self.income,
            "income_level": income_str_map[self.income_level],
            "income_group": self.income_group,
            "household_type": [
                household_type_reverse.get(idx, "") for idx in self.household_type_idx
            ],
            "elek_usage": self.elek_usage,
            "system_size": self.system_size,
            "base_cost": self.base_cost,
            "adopted": self.adopted,
            "adoption_time": self.adoption_time,
            "beta0": self.beta0,
            "social_weight": self.social_weight,
        }

        return pd.DataFrame(data)

    def get_slice(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Return a dict view of selected agent-level variables for given indices.
        """
        return {
            "ids": self.ids[indices],
            "income": self.income[indices],
            "adopted": self.adopted[indices],
            "V": self.V[indices],
            "S": self.S[indices],
            "P": self.P[indices],
        }
