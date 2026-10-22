"""Decision dashboard of the AM workshop twin.  Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from amtwin.model import apply_overrides, load_config, simulate  # noqa: E402
from amtwin.scenarios import flatten, summarise  # noqa: E402
from amtwin.surrogate import Surrogate, recommend  # noqa: E402

RES = ROOT / "results"
LABELS = {"printer": "Printers", "depowder": "Depowdering", "blast": "Bead blasting", "dye": "Dyeing",
          "finish": "Finishing", "qc": "Quality control", "pack": "Packing", "prep": "Job preparation",
          "operators": "Operators"}

st.set_page_config(page_title="AM workshop twin", layout="wide")
st.title("Additive manufacturing workshop: digital twin")
st.caption("Discrete-event model of the workshop. Parameters are assumptions until replaced by measured data.")

base = load_config()
r = base["resources"]
with st.sidebar:
    st.header("Scenario")
    ov = {
        "resources.printers": st.slider("Printers", 1, 6, r["printers"]),
        "resources.build_units": st.slider("Build units", 2, 10, r["build_units"]),
        "resources.depowder_stations": st.slider("Depowdering stations", 1, 3, r["depowder_stations"]),
        "resources.blast_cabinets": st.slider("Blast cabinets", 1, 3, r["blast_cabinets"]),
        "resources.finishing_benches": st.slider("Finishing benches", 1, 4, r["finishing_benches"]),
        "resources.qc_benches": st.slider("QC benches", 1, 3, r["qc_benches"]),
        "resources.operators_per_shift": st.slider("Operators per shift", 1, 6, r["operators_per_shift"]),
        "shifts.count": st.select_slider("Shifts per day", [1, 2, 3], base["shifts"]["count"]),
        "shifts.days_per_week": st.select_slider("Working days", [5, 6], base["shifts"]["days_per_week"]),
        "demand.orders_per_day": st.slider("Orders per day", 4.0, 14.0, float(base["demand"]["orders_per_day"]), 0.2),
    }
    reps = st.slider("Replications", 1, 10, 3)
    go = st.button("Run simulation", type="primary")


@st.cache_data(show_spinner=False)
def run(ov_items, reps):
    cfg = apply_overrides(base, dict(ov_items))
    return summarise(pd.DataFrame([flatten(simulate(cfg, 100 + i)) for i in range(reps)]))


tab_sim, tab_scen, tab_ai = st.tabs(["Simulation", "Scenario comparison", "AI pre-sizing"])

with tab_sim:
    ref = run(tuple(sorted({}.items())), reps)
    cur = run(tuple(sorted(ov.items())), reps) if go or "ran" in st.session_state else ref
    if go:
        st.session_state["ran"] = True
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean lead time (h)", f"{cur['lead_time_mean_h']:.0f}",
              f"{cur['lead_time_mean_h'] - ref['lead_time_mean_h']:+.0f} vs baseline", delta_color="inverse")
    c2.metric("Lead time P90 (h)", f"{cur['lead_time_p90_h']:.0f}")
    c3.metric("Parts shipped / week", f"{cur['parts_shipped_per_week']:.0f}",
              f"{cur['parts_shipped_per_week'] - ref['parts_shipped_per_week']:+.0f}")
    c4.metric("Orders still open at end", f"{cur['orders_open_at_end']:.0f}")
    util = {LABELS.get(k[5:], k[5:]): v for k, v in cur.items() if k.startswith("util_") and not k.endswith("_ci")}
    util = pd.Series(util).sort_values(ascending=False)
    st.subheader("Utilisation over available time")
    st.caption(f"First bottleneck suspect: **{util.index[0]}** ({util.iloc[0]:.0%}). Manual stations are measured "
               "over shift hours, printers and dyeing over calendar time.")
    st.bar_chart(util)
    waits = {LABELS.get(k[5:], k[5:]): v for k, v in cur.items() if k.startswith("wait_") and not k.endswith("_ci")}
    st.subheader("Mean waiting time before each step (h)")
    st.bar_chart(pd.Series(waits).sort_values(ascending=False))

with tab_scen:
    f = RES / "scenarios.csv"
    if f.exists():
        sc = pd.read_csv(f, index_col=0)
        cols = ["lead_time_mean_h", "lead_time_mean_h_ci", "lead_time_p90_h", "parts_shipped_per_week",
                "orders_open_at_end"]
        st.dataframe(sc[cols].round(1), width="stretch")
        st.caption("10 replications per scenario, ± is the 95% confidence half-width.")
    else:
        st.info("Run experiments/run_all.py to fill this table.")
    p = RES / "bottleneck_probe.csv"
    if p.exists():
        st.subheader("Bottleneck check: +1 unit of each resource")
        st.dataframe(pd.read_csv(p).round(1), width="stretch")

with tab_ai:
    d = RES / "surrogate_dataset.csv"
    if not d.exists():
        st.info("Run experiments/build_dataset.py first.")
    else:
        model = Surrogate().fit(pd.read_csv(d))
        demand = st.slider("Expected orders per day", 4.0, 12.0, 10.4, 0.2)
        target = st.slider("Target mean lead time (h)", 80, 240, 110)
        shortlist, n = recommend(model, demand, target, base, top=10)
        st.write(f"{n:,} configurations scored by the surrogate model in under a second.")
        st.dataframe(shortlist.round(1), width="stretch")
        st.warning("These are predictions. Confirm a shortlisted configuration with the Simulation tab before "
                   "any decision.")
