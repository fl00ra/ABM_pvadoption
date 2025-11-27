import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# from calibration results csv
CALIBRATED_PARAMS_CSV = "best_fit_params.csv"

# from CBS summary tables
CATEGORY_FILES = {
    "ownership": "250717_0500_20250716 - output 8969 - PV diffusion/category_counts_TypeEigenaar31Dec.csv",
    "household": "250717_0500_20250716 - output 8969 - PV diffusion/category_counts_TYPHH31Dec.csv",
    "woningtype": "250717_0500_20250716 - output 8969 - PV diffusion/category_counts_VBOWoningType31Dec.csv",
}

CONDITIONAL_FILES = {
    "household": "250717_0500_20250716 - output 8969 - PV diffusion/conditional_mean_TYPHH31Dec.csv",
    "ownership": "250717_0500_20250716 - output 8969 - PV diffusion/conditional_mean_TypeEigenaar31Dec.csv",
    "woningtype": "250717_0500_20250716 - output 8969 - PV diffusion/conditional_mean_VBOWoningType31Dec.csv",
}

HISTOGRAM_FILES = {
    "income": "250717_0500_20250716 - output 8969 - PV diffusion/histogram_GESTINKH31Dec.csv",
    "elek": "250717_0500_20250716 - output 8969 - PV diffusion/histogram_ELEK.csv",
    "woz": "250717_0500_20250716 - output 8969 - PV diffusion/histogram_WOZWaardeObjectBAG31Dec.csv",
}

CONTINGENCY_PV_FILES = {
    "ownership": "250717_0500_20250716 - output 8969 - PV diffusion/contingency_PV_TypeEigenaar31Dec.csv",
    "household": "250717_0500_20250716 - output 8969 - PV diffusion/contingency_PV_TYPHH31Dec.csv",
    "woning": "250717_0500_20250716 - output 8969 - PV diffusion/contingency_PV_VBOWoningType31Dec.csv",
}

CORRELATION_FILE = "250717_0500_20250716 - output 8969 - PV diffusion/pearson_correlation_2019.csv"

HOUSEHOLD_NL_EN = {
    "Eenpersoonshuishouden": "single",
    "Gehuwd paar zonder kinderen": "couple no kids",
    "Gehuwd paar met kinderen": "couple with kids",
    "Niet-gehuwd paar zonder kinderen": "cohabiting no kids",
    "Niet-gehuwd paar met kinderen": "cohabiting with kids",
    "Eenouderhuishouden": "single parent",
    "Overig huishouden": "other",
}

OWNERSHIP_NL_EN = {
    "EigenaarGebruiker": "private owner",
    "Woningcorporatie": "housing corporation",
    "Verhuurder anders dan woningcorporatie": "institutional investor",
    "Onbekend": "unknown",
}

WONINGTYPE_NL_EN = {
    "Tussenwoning": "mid-terrace",
    "Hoekwoning": "corner house",
    "Twee-onder-een-kapwoning": "semi-detached",
    "Vrijstaande woning": "detached",
    "Meergezinswoning": "apartment",
    "Eengezinswoning onbekend": "unknown",
}


def _load_medoid_woning_beta0(csv_path: str) -> dict:
    """
    Load dwelling-type-specific beta0 values from the calibrated CSV.
    """
    df = pd.read_csv(csv_path)
    s = df.set_index("param")["value"]

    keys = [
        "beta0::apartment",
        "beta0::detached",
        "beta0::semi-detached",
        "beta0::mid-terrace",
        "beta0::corner house",
        "beta0::unknown",
    ]

    beta_map: dict[str, float] = {}
    for k in keys:
        label = k.split("::", 1)[1]
        beta_map[label] = float(s[k])

    return beta_map


class DataLoader:
    """
    Loader for CBS-derived tables and synthetic household generation.
    """

    def __init__(self, base_dir: str, std_ratio: float = 0.3, year: int = 2019) -> None:
        self.base_dir = base_dir
        self.std_ratio = std_ratio
        self.year = year

        self._load_all()

        beta0_path = os.path.join(self.base_dir, CALIBRATED_PARAMS_CSV)
        self.beta0_woning_map = _load_medoid_woning_beta0(beta0_path)

    def _load_csv(self, relative_path: str) -> pd.DataFrame:
        full = os.path.join(self.base_dir, relative_path)
        return pd.read_csv(full)

    def _load_all(self) -> None:
        self._load_histograms()
        self._load_category_counts()
        self._load_conditional_means()
        self._load_contingency_tables()
        self._load_correlation_matrix()

    def _load_histograms(self) -> None:
        """Load histograms for income, electricity use, and property value."""
        self.hist_data: dict[str, dict] = {}
        for var, path in HISTOGRAM_FILES.items():
            df = self._load_csv(path)

            col_name = df.columns[1]
            year_col = str(self.year)

            df = df[[col_name, year_col]]
            df.columns = ["bin", "count"]
            self.hist_data[var] = self._process_histogram(df)

    def _process_histogram(self, df: pd.DataFrame) -> dict:
        """
        Convert a CBS-style histogram table into bins, centers, and probabilities.
        """
        bins: list[tuple[float, float]] = []
        counts: list[float] = []

        for _, row in df.iterrows():
            label = str(row["bin"])
            count = float(row["count"])

            if "<" in label:
                upper = float(label.replace("<", "").strip())
                bins.append((0.0, upper))
            elif "-" in label:
                low_str, high_str = label.split("-")
                low = float(low_str)
                high = float(high_str)
                bins.append((low, high))
            elif ">" in label:
                low = float(label.replace(">", "").strip())
                bins.append((low, low + 100000.0))
            else:
                continue

            counts.append(count)

        centers = [0.5 * (a + b) for (a, b) in bins]
        probs = np.array(counts, dtype=float)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones_like(probs, dtype=float) / len(probs)

        return {
            "bins": bins,
            "centers": centers,
            "probs": probs,
        }

    def _load_category_counts(self) -> None:
        """Load prior distributions over categorical variables."""
        self.category_priors: dict[str, dict] = {}
        for var, path in CATEGORY_FILES.items():
            df = self._load_csv(path)

            if "Unnamed: 0" in df.columns:
                df = df.set_index("Unnamed: 0")
            elif df.columns[0] == "":
                df = df.set_index(df.columns[0])

            row = df.loc[self.year]
            total = row.sum()
            if total > 0:
                self.category_priors[var] = (row / total).to_dict()

    def _load_conditional_means(self) -> None:
        """Load conditional means of numeric variables by categories."""
        self.cond_means: dict[str, dict] = {}
        for var, path in CONDITIONAL_FILES.items():
            df = self._load_csv(path)
            df = df[df["year"] == self.year].drop(columns=["year"])
            df = df.set_index(df.columns[0])
            self.cond_means[var] = df.to_dict(orient="index")

    def _load_contingency_tables(self) -> None:
        """Load contingency tables for PV adoption by category."""
        self.pv_probs: dict[str, dict] = {}
        for var, path in CONTINGENCY_PV_FILES.items():
            df = self._load_csv(path)

            df = df[df["year"] == self.year]
            drop_cols = ["Unnamed: 0", "year", "PV_binary"]
            drop_cols = [c for c in drop_cols if c in df.columns]

            df0 = df[df["PV_binary"] == 0].drop(columns=drop_cols).sum()
            df1 = df[df["PV_binary"] == 1].drop(columns=drop_cols).sum()

            total = df0 + df1
            with np.errstate(divide="ignore", invalid="ignore"):
                probs = df1 / total
            self.pv_probs[var] = probs.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_dict()

    def _load_correlation_matrix(self) -> None:
        """Load Pearson correlation matrix across numeric variables."""
        df = self._load_csv(CORRELATION_FILE)
        self.corr_matrix = df.set_index(df.columns[0]).astype(float)

    def _get_corr_col_name(self, var: str) -> str:
        """Map generic variable names to correlation matrix column names."""
        mapping = {
            "income": "GESTINKH31Dec",
            "elek": "ELEK",
            "woz": "WOZWaardeObjectBAG31Dec",
            "size": "VBOOppervlakte31Dec",
        }
        return mapping[var]

    def sample_numeric(
        self,
        var: str,
        conditioned_on: str | None = None,
        value: float | None = None,
    ) -> float:
        """
        Sample a numeric variable from its histogram, optionally
        re-weighted by a correlation-based conditional distribution.
        """
        data = self.hist_data[var]
        centers = np.array(data["centers"])
        probs = np.array(data["probs"])

        if conditioned_on is not None and value is not None:
            cond_data = self.hist_data[conditioned_on]

            var_col = self._get_corr_col_name(var)
            cond_col = self._get_corr_col_name(conditioned_on)

            corr = float(self.corr_matrix.loc[cond_col, var_col])

            cond_centers = np.array(cond_data["centers"])
            cond_probs = np.array(cond_data["probs"])

            cond_mean = np.dot(cond_centers, cond_probs)
            cond_std = np.sqrt(np.dot((cond_centers - cond_mean) ** 2, cond_probs))

            if cond_std > 0:
                target_z = (value - cond_mean) / cond_std

                var_mean = np.dot(centers, probs)
                var_std = np.sqrt(np.dot((centers - var_mean) ** 2, probs))

                if var_std > 0:
                    z_vals = (centers - var_mean) / var_std
                    expected_z = corr * target_z
                    weights = probs * np.exp(-0.5 * (z_vals - expected_z) ** 2)
                    total_w = weights.sum()
                    if total_w > 0:
                        probs = weights / total_w

        return float(np.random.choice(centers, p=probs))

    def infer_category(
        self,
        category_type: str,
        income: float,
        elek: float,
        woz: float,
    ) -> str:
        """
        Infer a categorical variable using naive Bayes with Gaussian likelihoods.
        """
        prior = self.category_priors[category_type]
        cond = self.cond_means[category_type]

        posterior: dict[str, float] = {}
        for cat, prior_p in prior.items():
            like = 1.0
            for var, obs in zip(
                ["GESTINKH31Dec", "ELEK", "WOZWaardeObjectBAG31Dec"],
                [income, elek, woz],
            ):
                if cat not in cond or var not in cond[cat]:
                    continue
                mean = cond[cat][var]
                std = max(self.std_ratio * mean, 1e-3)
                like *= norm.pdf(obs, loc=mean, scale=std)
            posterior[cat] = like * prior_p

        total = sum(posterior.values())
        if total <= 0:
            cats = list(prior.keys())
            probs = np.array(list(prior.values()), dtype=float)
            probs = probs / probs.sum()
            return str(np.random.choice(cats, p=probs))

        for k in posterior:
            posterior[k] /= total

        cats = list(posterior.keys())
        probs = np.array([posterior[c] for c in cats], dtype=float)
        probs = probs / probs.sum()

        return str(np.random.choice(cats, p=probs))

    def get_pv_prob(self, ownership: str, household: str, woning: str) -> float:
        """
        Combine PV adoption probabilities from three contingency tables.
        """
        p1 = self.pv_probs["ownership"].get(ownership, 0.0)
        p2 = self.pv_probs["household"].get(household, 0.0)
        p3 = self.pv_probs["woning"].get(woning, 0.0)
        return float(np.mean([p1, p2, p3]))

    def _beta0_from_medoid(self, agent_data: dict) -> float:
        """
        Map dwelling type to its calibrated beta0.
        """
        w = agent_data["woning_type"]
        return float(self.beta0_woning_map[w])

    def generate_agent(self) -> dict:
        """
        Generate a single synthetic agent with CBS-consistent attributes.
        """
        income = self.sample_numeric("income")
        elek = self.sample_numeric("elek", conditioned_on="income", value=income)
        woz = self.sample_numeric("woz", conditioned_on="income", value=income)

        hh_type = self.infer_category("household", income, elek, woz)
        ownership = self.infer_category("ownership", income, elek, woz)
        woning = self.infer_category("woningtype", income, elek, woz)

        ownership_en = OWNERSHIP_NL_EN.get(ownership, "unknown")
        adopted = (
            np.random.rand() < 0.19 if ownership_en == "private owner" else False
        )

        agent_data = {
            "income": income,
            "elek_usage": elek,
            "woz_value": woz,
            "household_type": HOUSEHOLD_NL_EN.get(hh_type, "other"),
            "ownership_status": ownership_en,
            "woning_type": WONINGTYPE_NL_EN.get(woning, "unknown"),
            "adopted": adopted,
        }

        agent_data["beta0"] = self._beta0_from_medoid(agent_data)
        return agent_data


def load_amsterdam_data(n_agents: int, policy=None) -> list[dict]:
    """
    Generate a list of synthetic agents for Amsterdam.
    """
    _ = policy
    loader = DataLoader(base_dir=".", year=2019)
    return [loader.generate_agent() for _ in range(n_agents)]
