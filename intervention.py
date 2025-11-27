import numpy as np

elec_price_const = ...


class Policy:
    def __init__(
        self,
        behavior_mode: str = "universal_subsidy",
        enable_feed_change: bool = True,
        fit_mode: str = "declining_net_metering",
        fit_params: dict | None = None,
    ):
        """
        Policy object controlling economic incentives and pricing.

        Parameters:
            behavior_mode: Policy regime, e.g. "no_policy", "universal_subsidy",
                "income_targeted_subsidy", "tiered_pricing", "open_to_tenants".
            fit_mode: Net-metering regime:
                "no_net_metering", "fixed_net_metering", "declining_net_metering".
            fit_params: Optional overrides for the declining schedule.
        """
        self.behavior_mode = behavior_mode
        self.enable_feed_change = enable_feed_change

        normalized_fit_mode = fit_mode
        if normalized_fit_mode == "fixed_retail":
            normalized_fit_mode = "fixed_net_metering"

        allowed_modes = {
            "no_net_metering",
            "fixed_net_metering",
            "declining_net_metering",
        }
        if normalized_fit_mode not in allowed_modes:
            normalized_fit_mode = "no_net_metering"

        if (not enable_feed_change) and normalized_fit_mode == "declining_net_metering":
            normalized_fit_mode = "fixed_net_metering"

        self.fit_mode = normalized_fit_mode

        self.policy_params = {
            "universal_subsidy": {
                "discount_rate": 0.15,
                "start_time": 1,
                "end_time": None,
            },
            "income_targeted_subsidy": {
                "low_rate": 0.20,
                "mid_rate": 0.15,
                "high_rate": 0.10,
                "start_time": 1,
                "end_time": None,
            },
            "tiered_pricing": {
                "level_1_quantile": 75,
                "level_2_quantile": 90,
                "level_1_range": (1.25, 1.5),
                "level_2_range": (1.75, 2.0),
            },
            "open_to_tenants": {
                "note": "Allow non-owner households (e.g. tenants) to adopt PV."
            },
        }

        self.fit_params = {
            "start_t": 10,
            "duration": 5,
            "floor": 0.001,
        }
        if fit_params is not None:
            self.fit_params.update(fit_params)

    def _net_metering_share_at_t(self, t):
        """
        Net-metering share in [0, 1] at absolute timestep t.
        """
        mode = self.fit_mode
        t_arr = np.asarray(t, dtype=int)

        if mode == "no_net_metering":
            return np.zeros_like(t_arr, dtype=float)

        if mode == "fixed_net_metering":
            return np.ones_like(t_arr, dtype=float)

        if mode == "declining_net_metering":
            start_t = int(self.fit_params.get("start_t", 10))
            duration = int(self.fit_params.get("duration", 5))
            end_t = start_t + duration

            share = np.zeros_like(t_arr, dtype=float)
            before = t_arr < start_t
            share[before] = 1.0

            if duration > 0:
                within = (t_arr >= start_t) & (t_arr < end_t)
                if np.any(within):
                    frac = (t_arr[within] - start_t) / float(duration)
                    share[within] = np.clip(1.0 - frac, 0.0, 1.0)

            return share

        return np.zeros_like(t_arr, dtype=float)

    def get_net_metering_share_expectation(self, decision_t, future_t):
        """
        Expected net-metering share at future_t for an agent deciding at decision_t.
        """
        _ = decision_t
        return self._net_metering_share_at_t(future_t)

    def _fit_at_t(self, t, current_elec_price=None):
        """
        Feed-in tariff (FIT) at absolute timestep t.

        If current_elec_price is an array, its mean is used as base price.
        """
        if current_elec_price is None:
            base_price = float(elec_price_const)
        else:
            arr = np.asarray(current_elec_price, dtype=float)
            if arr.size > 0:
                base_price = float(arr.mean())
            else:
                base_price = float(elec_price_const)

        mode = self.fit_mode
        t_arr = np.asarray(t, dtype=int)

        if mode == "no_net_metering":
            return np.zeros_like(t_arr, dtype=float)

        if mode == "fixed_net_metering":
            return np.full_like(t_arr, base_price, dtype=float)

        if mode == "declining_net_metering":
            start_t = int(self.fit_params.get("start_t", 10))
            duration = int(self.fit_params.get("duration", 5))
            floor = float(self.fit_params.get("floor", 0.001))
            end_t = start_t + duration

            if duration <= 0:
                return np.full_like(t_arr, floor, dtype=float)

            out = np.full_like(t_arr, base_price, dtype=float)

            before = t_arr < start_t
            during = (t_arr >= start_t) & (t_arr <= end_t)
            after = t_arr > end_t

            frac = (t_arr - start_t) / float(duration)
            frac = np.clip(frac, 0.0, 1.0)

            out[during] = base_price * (1.0 - frac[during]) + floor * frac[during]
            out[after] = floor
            return out

        return np.zeros_like(t_arr, dtype=float)

    def get_feed_in_tariff(self, timestep, current_elec_price=None):
        """
        Realized FIT at a given timestep.
        """
        t_arr = np.asarray(timestep, dtype=int)
        return self._fit_at_t(t_arr, current_elec_price)

    def get_feed_in_tariff_expectation(
        self,
        decision_t,
        future_t,
        current_elec_price=None,
    ):
        """
        Expected FIT at future_t for an agent deciding at decision_t.
        """
        _ = decision_t
        f = np.asarray(future_t, dtype=int)
        return self._fit_at_t(f, current_elec_price)

    def block_bill_and_avg_price_fixed(self, elek_usage):
        """
        Increasing-block tariff for annual electricity consumption.

        Blocks (kWh per year):
            [0, 3000]    : base
            (3000, 5000] : base + 0.05
            > 5000       : base + 0.10
        """
        base = float(elec_price_const)
        limits = np.array([3000.0, 5000.0], dtype=np.float64)
        add_tax = np.array([0.00, 0.05, 0.10], dtype=np.float64)
        rates = base + add_tax

        u = np.asarray(elek_usage, dtype=np.float64)

        tier1 = np.minimum(u, limits[0])
        tier2 = np.clip(u - limits[0], 0, limits[1] - limits[0])
        tier3 = np.clip(u - limits[1], 0, None)

        bill = tier1 * rates[0] + tier2 * rates[1] + tier3 * rates[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_price = np.where(u > 0, bill / u, base)
        return bill, avg_price

    def get_congestion_price_vectorized(self, elek_usage, timestep):
        """
        Map yearly usage to a single bracket-specific price.

        For NPV or billing, use block_bill_and_avg_price_fixed instead.
        """
        _ = timestep

        u = np.asarray(elek_usage, dtype=np.float64)
        base = float(elec_price_const)

        if self.behavior_mode != "tiered_pricing":
            return np.full_like(u, base, dtype=np.float64)

        limits = np.array([3000.0, 5000.0], dtype=np.float64)
        add_tax = np.array([0.00, 0.05, 0.10], dtype=np.float64)
        bracket_rates = base + add_tax

        price = np.full_like(u, bracket_rates[0], dtype=np.float64)
        price[(u > limits[0]) & (u <= limits[1])] = bracket_rates[1]
        price[u > limits[1]] = bracket_rates[2]
        return price

    def get_effective_cost(self, base_costs, timestep, income_group=None):
        """
        Apply policy-specific subsidies to baseline PV investment costs.
        """
        mode = self.behavior_mode

        if mode == "universal_subsidy":
            params = self.policy_params["universal_subsidy"]
            start = params.get("start_time", 1)
            end = params.get("end_time", None)
            if timestep >= start and (end is None or timestep <= end):
                return base_costs * (1.0 - params["discount_rate"])
            return base_costs

        if mode == "income_targeted_subsidy":
            params = self.policy_params["income_targeted_subsidy"]
            start = params.get("start_time", 1)
            end = params.get("end_time", None)

            if timestep < start or (end is not None and timestep > end):
                return base_costs

            if income_group is None:
                return base_costs

            income_group = np.asarray(income_group, dtype=int)
            low_r = float(params["low_rate"])
            mid_r = float(params["mid_rate"])
            high_r = float(params["high_rate"])

            discounts = np.zeros_like(base_costs, dtype=float)
            discounts[income_group == 0] = low_r
            discounts[income_group == 1] = mid_r
            discounts[income_group == 2] = high_r

            return base_costs * (1.0 - discounts)

        return base_costs
