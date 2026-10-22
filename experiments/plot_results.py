"""Figures of the README from the files in results/."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from amtwin.surrogate import Surrogate  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "results" / "figures"
INK, MUTED, ACCENT, GOOD = "#1d2a33", "#8a99a6", "#e07b39", "#2f7f6f"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def scenarios():
    sc = pd.read_csv(RES / "scenarios.csv", index_col=0)
    sc = sc.drop(index=[i for i in sc.index if i.startswith("Demand")])
    sc = sc.sort_values("lead_time_mean_h")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = [MUTED if n == "Baseline" else (ACCENT if "printer" in n else GOOD) for n in sc.index]
    ax.barh(sc.index, sc["lead_time_mean_h"], xerr=sc["lead_time_mean_h_ci"], color=colors, capsize=3)
    ax.set_xlabel("Mean lead time, order to shipment (h), 10 replications, 95% CI")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_scenarios.png", dpi=150)


def utilisation():
    sc = pd.read_csv(RES / "scenarios.csv", index_col=0)
    b = sc.loc["Baseline"]
    u = {c[5:]: b[c] for c in sc.columns if c.startswith("util_") and not c.endswith("_ci")}
    names = {"operators": "Operators", "blast": "Bead blasting", "qc": "Quality control", "finish": "Finishing",
             "printer": "Printers", "depowder": "Depowdering", "dye": "Dyeing", "prep": "Job preparation",
             "pack": "Packing"}
    u = pd.Series({names[k]: v for k, v in u.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(u.index, u.values * 100, color=[ACCENT if v == u.max() else MUTED for v in u.values])
    ax.set_xlabel("Utilisation over available time (%), baseline")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_utilisation.png", dpi=150)


def surrogate():
    df = pd.read_csv(RES / "surrogate_dataset.csv")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    cut = int(len(df) * 0.75)
    train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    pred = Surrogate().fit(train).predict(test)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.loglog(test["lead_time_mean_h"], pred["lead_pred_h"], "o", ms=4, color=GOOD, alpha=0.8)
    lim = [70, 1200]
    ax.plot(lim, lim, color=MUTED, lw=1)
    ax.set_xlabel("Simulated mean lead time (h)")
    ax.set_ylabel("Surrogate prediction (h)")
    ax.set_title("100 configurations never seen in training", fontsize=9, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_surrogate.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    scenarios()
    utilisation()
    surrogate()
    print(json.dumps(sorted(p.name for p in FIG.glob("*.png"))))
