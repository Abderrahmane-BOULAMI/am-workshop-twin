"""Machine-learning surrogate of the simulation, used to screen thousands of configurations in seconds.

1. Sample workshop configurations (random design), simulate each once.
2. Train gradient boosting to predict lead time and throughput from the configuration.
3. Score every configuration of a grid with the surrogate, keep the cheapest that meets the target.
4. Re-simulate the shortlist with replications: the surrogate proposes, the simulation decides.
"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .model import apply_overrides, load_config, simulate

SPACE = {  # parameter -> candidate values
    "resources.printers": [2, 3, 4, 5],
    "resources.build_units": [3, 4, 5, 6, 8],
    "resources.depowder_stations": [1, 2],
    "resources.blast_cabinets": [1, 2, 3],
    "resources.finishing_benches": [1, 2, 3],
    "resources.qc_benches": [1, 2],
    "resources.operators_per_shift": [2, 3, 4, 5],
    "shifts.count": [1, 2, 3],
    "shifts.days_per_week": [5, 6],
    "demand.orders_per_day": [6.0, 8.0, 10.4, 12.0],
}
FEATURES = [k.split(".")[-1] for k in SPACE]


def sample_design(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()} for _ in range(n)]


def _sim(args):
    base, ov, seed = args
    k = simulate(apply_overrides(base, ov), seed)
    return {**{key.split(".")[-1]: v for key, v in ov.items()},
            "lead_time_mean_h": k["lead_time_censored_mean_h"], "parts_shipped_per_week": k["parts_shipped_per_week"],
            "orders_open_at_end": k["orders_open_at_end"]}


def build_dataset(n: int = 400, seed: int = 0, workers: int = 2, horizon_weeks: int = 9) -> pd.DataFrame:
    base = apply_overrides(load_config(), {"horizon_weeks": horizon_weeks})
    design = sample_design(n, seed)
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_sim, [(base, ov, 1000 + i) for i, ov in enumerate(design)], chunksize=4))
    return pd.DataFrame(rows)


class Surrogate:
    def __init__(self):
        self.lead = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)
        self.tput = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0)

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].to_numpy(float)
        self.lead.fit(X, np.log(df["lead_time_mean_h"].to_numpy()))
        self.tput.fit(X, df["parts_shipped_per_week"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURES].to_numpy(float)
        return pd.DataFrame({"lead_pred_h": np.exp(self.lead.predict(X)), "tput_pred": self.tput.predict(X)},
                            index=df.index)


def evaluate(df: pd.DataFrame, test_share: float = 0.25, seed: int = 0) -> dict:
    """Hold-out accuracy of the surrogate on configurations it never saw."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_share))
    train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    pred = Surrogate().fit(train).predict(test)
    y = test["lead_time_mean_h"].to_numpy()
    return {"n_train": len(train), "n_test": len(test),
            "lead_r2": float(r2_score(np.log(y), np.log(pred["lead_pred_h"]))),
            "lead_mape": float(np.mean(np.abs(pred["lead_pred_h"].to_numpy() - y) / y)),
            "tput_r2": float(r2_score(test["parts_shipped_per_week"], pred["tput_pred"])),
            "tput_mae": float(mean_absolute_error(test["parts_shipped_per_week"], pred["tput_pred"]))}


def yearly_cost(row: dict, cfg: dict) -> float:
    """Indicative yearly cost in kEUR: equipment written off over 5 years + operator cost."""
    c, r = cfg["costs"], row
    equipment = (c["printer"] * r["printers"] + c["build_unit"] * r["build_units"]
                 + c["depowder_station"] * r["depowder_stations"] + c["blast_cabinet"] * r["blast_cabinets"]
                 + c["finishing_bench"] * r["finishing_benches"] + c["qc_bench"] * r["qc_benches"])
    labour = c["operator_per_shift_year"] * r["operators_per_shift"] * r["count"] * r["days_per_week"] / 5
    return equipment / 5 + labour


def grid(demand: float) -> pd.DataFrame:
    keys = [k for k in SPACE if k != "demand.orders_per_day"]
    rows = [dict(zip([k.split(".")[-1] for k in keys], combo)) for combo in itertools.product(*[SPACE[k] for k in keys])]
    df = pd.DataFrame(rows)
    df["orders_per_day"] = demand
    return df


def recommend(model: Surrogate, demand: float, max_lead_h: float, cfg: dict | None = None,
              top: int = 5) -> pd.DataFrame:
    """Cheapest configurations predicted to meet the lead-time target at the given demand."""
    cfg = cfg or load_config()
    g = grid(demand)
    g = g.join(model.predict(g))
    g["cost_keur_year"] = [yearly_cost(r, cfg) for r in g.to_dict("records")]
    ok = g[g["lead_pred_h"] <= max_lead_h]
    return ok.sort_values("cost_keur_year").head(top).reset_index(drop=True), len(g)


def to_overrides(row: dict) -> dict:
    return {k: (float(row[k.split(".")[-1]]) if k.startswith("demand") else int(row[k.split(".")[-1]]))
            for k in SPACE}
