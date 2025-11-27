import os
import json
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from model import ABM
from config import n_agents, n_steps, beta
from network import build_multiple_networks
from data_loader import load_amsterdam_data
from agent_state import AgentArray


BASE_SEED: int = 1
N_SEEDS: int = 20
MAX_WORKERS: int = int(os.environ.get("PV_MAX_WORKERS", "4"))
ROW_GROUP_SIZE: int = int(os.environ.get("PV_ROW_GROUP", "100000"))

INTRA_RATIOS: List[float] = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]

PHASEOUT_PARAMS: Dict[str, float] = {
    "start_t": 10,
    "duration": 5,
    "floor": 0.001,
}

SCENARIOS: List[Dict] = [
    {
        "name": "baseline",
        "behavior_mode": "no_policy",
        "fit_mode": "no_net_metering",
        "fit_params": None,
    },
    {
        "name": "universal_fixed_nm",
        "behavior_mode": "universal_subsidy",
        "fit_mode": "fixed_net_metering",
        "fit_params": None,
    },
    {
        "name": "targeted_fixed_nm",
        "behavior_mode": "income_targeted_subsidy",
        "fit_mode": "fixed_net_metering",
        "fit_params": None,
    },
    {
        "name": "universal_phaseout_nm",
        "behavior_mode": "universal_subsidy",
        "fit_mode": "declining_net_metering",
        "fit_params": PHASEOUT_PARAMS,
    },
    {
        "name": "targeted_phaseout_nm",
        "behavior_mode": "income_targeted_subsidy",
        "fit_mode": "declining_net_metering",
        "fit_params": PHASEOUT_PARAMS,
    },
    {
        "name": "tiered_phaseout_nm",
        "behavior_mode": "tiered_pricing",
        "fit_mode": "declining_net_metering",
        "fit_params": PHASEOUT_PARAMS,
    },
    {
        "name": "tiered_fixed_nm",
        "behavior_mode": "tiered_pricing",
        "fit_mode": "fixed_net_metering",
        "fit_params": None,
    },
]

OUT_ROOT: Path = Path("/Volumes") / "One Touch" / "pv_results"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PQT_COMPRESSION_CANDIDATES = ["zstd", "snappy", "gzip"]

ANCHOR_PATH = "anchors/V_anchor_asinh_no_policy.json"
with open(ANCHOR_PATH, "r") as f:
    _A = json.load(f)

V_ANCHOR_SPEC: Dict[str, float] = {
    "s": float(_A["s"]),
    "mu_star": float(_A["mu_star"]),
    "sd_star": float(_A["sd_star"]),
}


def now_run_id() -> str:
    """
    Build a short run identifier based on current date (YYYY-MM-DD).
    """
    return datetime.now().strftime("%Y-%m-%d")


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_parquet_compression() -> str:
    """
    Choose a Parquet compression codec, preferring common options.
    """
    for c in PQT_COMPRESSION_CANDIDATES:
        try:
            if pa.Codec.is_available(c):
                return c
        except Exception:
            continue
    return "snappy"


def parquet_path(
    root: Path,
    run_id: str,
    seed: int,
    policy: str,
    homophily: float,
    timestep: int,
) -> Path:
    """
    Build a partitioned Parquet path for a given run component.
    """
    return (
        root
        / f"run_id={run_id}"
        / f"seed={seed}"
        / f"policy={policy}"
        / f"h={homophily:.2f}"
        / f"part-t={timestep:03d}.parquet"
    )


def write_parquet_file(
    df: pd.DataFrame,
    path: Path,
    *,
    compression: str,
    row_group_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression=compression,
        write_statistics=True,
        row_group_size=row_group_size,
        use_dictionary=True,
    )


def stream_agent_history_to_parquet(
    model: ABM,
    *,
    out_root: Path,
    run_id: str,
    seed: int,
    policy: str,
    homophily: float,
    compression: str,
    row_group_size: int,
) -> List[str]:
    """
    Stream the agent_history of a model to partitioned Parquet files.
    """
    written: List[str] = []

    for i, df in enumerate(model.agent_history):
        if df is None or len(df) == 0:
            continue

        for bcol in ["adopted", "is_new_adopter"]:
            if bcol in df.columns:
                df[bcol] = (
                    df[bcol]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1", "t", "yes"])
                    .astype("int8")
                )

        if "policy" not in df.columns:
            df = df.copy()
            df["policy"] = policy
            df["homophily"] = homophily
            df["seed"] = seed

        ts = int(df["timestep"].iloc[0]) if "timestep" in df.columns else i
        p = parquet_path(out_root, run_id, seed, policy, homophily, ts)

        write_parquet_file(
            df,
            p,
            compression=compression,
            row_group_size=row_group_size,
        )
        written.append(str(p))
        model.agent_history[i] = None

    model.agent_history.clear()
    return written


def run_simulation_to_parquet(
    behavior_mode: str,
    fit_mode: str,
    fit_params: Dict,
    scenario_name: str,
    intra_ratio: float,
    adjacency_matrix: np.ndarray,
    agent_data: Dict,
    group_ids: np.ndarray,
    seed: int,
    out_root: Path,
    run_id: str,
    compression: str,
    row_group_size: int,
) -> List[str]:
    """
    Run one scenario on a given network and write the full agent history.
    """
    model = ABM(
        n_agents=n_agents,
        beta=beta,
        behavior_mode=behavior_mode,
        enable_feed_change=True,
        fit_mode=fit_mode,
        fit_params=fit_params,
        load_data=lambda x, policy=None: agent_data,
        adjacency_matrix=adjacency_matrix,
        group_ids=group_ids,
        external_V_anchor=V_ANCHOR_SPEC,
        seed=seed,
        run_label=scenario_name,
    )
    model.run(n_steps)

    policy_label = scenario_name

    return stream_agent_history_to_parquet(
        model,
        out_root=out_root,
        run_id=run_id,
        seed=seed,
        policy=policy_label,
        homophily=intra_ratio,
        compression=compression,
        row_group_size=row_group_size,
    )


def worker_run(
    seed: int,
    intra_ratios: List[float],
    scenarios: List[Dict],
    out_root: str,
    run_id: str,
    compression: str,
    row_group_size: int,
) -> Dict:
    """
    Run all scenarios for one seed and write outputs under one shard.
    """
    set_global_seed(seed)

    out_root_p = Path(out_root)
    written_all: List[str] = []

    print(f"[Seed {seed}] Loading agent data…", flush=True)
    agent_data = load_amsterdam_data(n_agents)
    dummy_agents = AgentArray(n_agents, agent_data, None)

    print(f"[Seed {seed}] Building networks for intra_ratios={intra_ratios}", flush=True)
    networks, _, group_ids = build_multiple_networks(
        dummy_agents,
        intra_ratios=intra_ratios,
    )

    for intra_ratio in intra_ratios:
        adjacency_matrix = networks[intra_ratio]
        for sc in scenarios:
            behavior_mode = sc["behavior_mode"]
            fit_mode = sc["fit_mode"]
            fit_params = sc.get("fit_params", None)
            scenario_name = sc["name"]

            print(
                f"[Seed {seed}] Running: scenario={scenario_name}, "
                f"policy={behavior_mode}, fit={fit_mode}, h={intra_ratio}",
                flush=True,
            )
            paths = run_simulation_to_parquet(
                behavior_mode=behavior_mode,
                fit_mode=fit_mode,
                fit_params=fit_params,
                scenario_name=scenario_name,
                intra_ratio=intra_ratio,
                adjacency_matrix=adjacency_matrix,
                agent_data=agent_data,
                group_ids=group_ids,
                seed=seed,
                out_root=out_root_p,
                run_id=run_id,
                compression=compression,
                row_group_size=row_group_size,
            )
            written_all.extend(paths)

    return {"seed": seed, "files": written_all}


def write_manifest(manifest_path: Path, run_id: str, shards: List[Dict]) -> None:
    """
    Write a JSON manifest describing all Parquet shards for this run.
    """
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "format": "parquet",
        "root": str(manifest_path.parent),
        "seeds": {str(s["seed"]): s["files"] for s in shards},
        "note": (
            "Query with DuckDB: "
            "SELECT * FROM read_parquet('<root>/run_id=<run_id>/**.parquet');"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    """
    Entry point for running all seeds, networks, and scenarios.
    """
    run_id = now_run_id()
    compression = pick_parquet_compression()

    print(f"\n[RUN] run_id={run_id}")
    print(f"[RUN] Output root: {OUT_ROOT}")
    print(f"[RUN] Seeds: BASE={BASE_SEED}, N={N_SEEDS}")
    print(f"[RUN] Parallel workers: {MAX_WORKERS}")
    print(f"[RUN] Parquet compression: {compression}")
    print(f"[RUN] Row group size: {ROW_GROUP_SIZE}\n")

    shards: List[Dict] = []
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]

    if MAX_WORKERS <= 1:
        for seed in seeds:
            res = worker_run(
                seed,
                INTRA_RATIOS,
                SCENARIOS,
                str(OUT_ROOT),
                run_id,
                compression,
                ROW_GROUP_SIZE,
            )
            shards.append(res)
    else:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [
                ex.submit(
                    worker_run,
                    seed,
                    INTRA_RATIOS,
                    SCENARIOS,
                    str(OUT_ROOT),
                    run_id,
                    compression,
                    ROW_GROUP_SIZE,
                )
                for seed in seeds
            ]
            for fut in as_completed(futures):
                shards.append(fut.result())

    manifest_path = OUT_ROOT / f"manifest_{run_id}.json"
    write_manifest(manifest_path, run_id, shards)

    print(f"\n[DONE] Wrote manifest: {manifest_path}")
    for s in sorted(shards, key=lambda x: x["seed"]):
        print(f"  - seed {s['seed']}: {len(s['files'])} files")


if __name__ == "__main__":
    main()
