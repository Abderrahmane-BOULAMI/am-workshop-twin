"""Scenario evaluation with replications, confidence intervals and bottleneck detection."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .model import apply_overrides, load_config, simulate

# What-if scenarios a workshop manager typically asks about. Values are overrides of the baseline.
SCENARIOS = {
    "Baseline": {},
    "+1 printer": {"resources.printers": 4},
    "+1 operator per shift": {"resources.operators_per_shift": 4},
    "3 shifts": {"shifts.count": 3},
    "Saturday shifts": {"shifts.days_per_week": 6},
    "+1 blast cabinet": {"resources.blast_cabinets": 2},
    "+1 blast cabinet and +1 operator": {"resources.blast_cabinets": 2, "resources.operators_per_shift": 4},
    "Demand +30%": {"demand.orders_per_day": 10.4},
    "Demand +30%, +1 blast, +1 operator": {"demand.orders_per_day": 10.4, "resources.blast_cabinets": 2,
                                          "resources.operators_per_shift": 4},
}

# One extra unit of each resource, one at a time: the resource whose increase helps most is the bottleneck.
CAPACITY_PROBES = {
    "printers": "resources.printers", "build_units": "resources.build_units",
    "depowder_stations": "resources.depowder_stations", "blast_cabinets": "resources.blast_cabinets",
    "dye_vats": "resources.dye_vats", "finishing_benches": "resources.finishing_benches",
    "qc_benches": "resources.qc_benches", "operators_per_shift": "resources.operators_per_shift",
}

KPI_COLS = ["lead_time_censored_mean_h", "lead_time_mean_h", "lead_time_p90_h", "parts_shipped_per_week", "orders_open_at_end"]


def _run(args):
    cfg, seed = args
    return simulate(cfg, seed)


def flatten(k: dict) -> dict:
    row = {c: k[c] for c in KPI_COLS + ["jobs_per_week", "mean_build_fill", "orders_unfinished_after_drain"]}
    row.update({f"util_{s}": v for s, v in k["utilisation"].items()})
    row.update({f"wait_{s}": v for s, v in k["mean_wait_h"].items()})
    return row


def replicate(cfg: dict, n: int = 10, workers: int = 2, seed0: int = 100) -> pd.DataFrame:
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_run, [(cfg, seed0 + i) for i in range(n)]))
    return pd.DataFrame([flatten(r) for r in res])


def summarise(df: pd.DataFrame) -> dict:
    """Mean and 95% confidence half-width (Student t approximated by 2.26 for n = 10)."""
    from scipy import stats
    n = len(df)
    t = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    out = {}
    for c in df.columns:
        out[c] = float(df[c].mean())
        out[c + "_ci"] = float(t * df[c].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return out


def run_scenarios(base: dict | None = None, scenarios: dict | None = None, n: int = 10) -> pd.DataFrame:
    base = base or load_config()
    rows = []
    for name, ov in (scenarios or SCENARIOS).items():
        s = summarise(replicate(apply_overrides(base, ov), n))
        s["scenario"] = name
        rows.append(s)
    return pd.DataFrame(rows).set_index("scenario")


def bottleneck_probe(base: dict | None = None, n: int = 10) -> pd.DataFrame:
    """+1 unit of each resource: gain in mean lead time and throughput versus baseline (common seeds)."""
    base = base or load_config()
    ref = summarise(replicate(base, n))
    rows = []
    for name, key in CAPACITY_PROBES.items():
        node = base
        for p in key.split(".")[:-1]:
            node = node[p]
        cur = node[key.split(".")[-1]]
        s = summarise(replicate(apply_overrides(base, {key: cur + 1}), n))
        rows.append({"resource": name, "from": cur, "to": cur + 1,
                     "lead_time_mean_h": s["lead_time_mean_h"],
                     "lead_time_gain_h": ref["lead_time_mean_h"] - s["lead_time_mean_h"],
                     "lead_time_gain_ci": s["lead_time_mean_h_ci"],
                     "throughput_gain": s["parts_shipped_per_week"] - ref["parts_shipped_per_week"]})
    return pd.DataFrame(rows).sort_values("lead_time_gain_h", ascending=False).reset_index(drop=True)


def utilisation_ranking(kpis_row: dict) -> list[tuple[str, float]]:
    """Resources ranked by utilisation over their available time (first one = first bottleneck suspect)."""
    u = {k[5:]: v for k, v in kpis_row.items() if k.startswith("util_") and not k.endswith("_ci")}
    return sorted(u.items(), key=lambda kv: -kv[1])
