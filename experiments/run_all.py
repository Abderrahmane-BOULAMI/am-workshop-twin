"""Reproduce every number in the README: scenarios, bottleneck probe, surrogate, G-code check.

    python experiments/build_dataset.py   # once, about 5 min
    python experiments/run_all.py         # about 6 min on 2 cores
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from amtwin.gcode import make_box_gcode, parse  # noqa: E402
from amtwin.model import apply_overrides, load_config  # noqa: E402
from amtwin.scenarios import bottleneck_probe, replicate, run_scenarios, summarise, utilisation_ranking  # noqa: E402
from amtwin.surrogate import Surrogate, evaluate, recommend, to_overrides  # noqa: E402

RES = ROOT / "results"


def main():
    t0 = time.time()
    RES.mkdir(exist_ok=True)
    base = load_config()
    summary = {}

    sc = run_scenarios(base, n=10)
    sc.to_csv(RES / "scenarios.csv")
    b = sc.loc["Baseline"].to_dict()
    summary["baseline"] = {k: b[k] for k in ["lead_time_mean_h", "lead_time_mean_h_ci", "lead_time_p90_h",
                                             "parts_shipped_per_week", "jobs_per_week", "mean_build_fill"]}
    summary["utilisation_ranking"] = utilisation_ranking(b)[:5]
    summary["scenarios"] = sc[["lead_time_mean_h", "lead_time_mean_h_ci", "lead_time_p90_h",
                               "parts_shipped_per_week", "orders_open_at_end"]].round(1).to_dict("index")

    probe = bottleneck_probe(base, n=10)
    probe.to_csv(RES / "bottleneck_probe.csv", index=False)
    summary["bottleneck_probe"] = probe.round(1).to_dict("records")

    df = pd.read_csv(RES / "surrogate_dataset.csv")
    summary["surrogate_eval"] = evaluate(df)
    model = Surrogate().fit(df)
    target = 110.0  # hours, order to shipment, at +30% demand (255 h if nothing changes)
    # propose with the surrogate, confirm by simulation; if nothing passes, ask for a stricter prediction
    checks, rounds = [], []
    for factor in (1.0, 0.85, 0.7, 0.55):
        shortlist, n_grid = recommend(model, demand=10.4, max_lead_h=target * factor, cfg=base, top=3)
        passed = False
        for row in shortlist.to_dict("records"):
            s = summarise(replicate(apply_overrides(base, to_overrides(row)), 10))
            ok = bool(s["lead_time_censored_mean_h"] <= target)
            passed |= ok
            checks.append({"round_threshold_h": round(target * factor, 1),
                           **{k: row[k] for k in ["printers", "build_units", "depowder_stations", "blast_cabinets",
                                                  "finishing_benches", "qc_benches", "operators_per_shift", "count",
                                                  "days_per_week"]},
                           "cost_keur_year": round(row["cost_keur_year"], 1),
                           "lead_pred_h": round(row["lead_pred_h"], 1),
                           "lead_sim_h": round(s["lead_time_censored_mean_h"], 1),
                           "lead_sim_ci": round(s["lead_time_censored_mean_h_ci"], 1), "meets_target": ok})
        rounds.append({"threshold_h": round(target * factor, 1), "passed": passed})
        if passed:
            break
    best = min((c for c in checks if c["meets_target"]), key=lambda c: c["cost_keur_year"], default=None)
    summary["recommendation"] = {"demand_orders_per_day": 10.4, "target_lead_h": target, "grid_size": n_grid,
                                 "simulated_configurations": len(checks), "rounds": rounds, "best": best,
                                 "checked": checks}
    pd.DataFrame(checks).to_csv(RES / "recommendation_check.csv", index=False)

    g = parse(make_box_gcode(40, 10, 0.2, 3000), accel_mm_s2=1000)
    summary["gcode_box_check"] = g.as_dict()
    summary["runtime_s"] = round(time.time() - t0)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
