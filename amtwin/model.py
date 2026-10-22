"""Discrete-event model of an additive manufacturing workshop.

Flow of one order: parts wait for nesting -> a build job is prepared -> automatic load -> printing ->
cooling in its build unit -> depowdering -> per part: bead blasting, dyeing (batch, optional),
finishing, quality control (rejects go back to nesting) -> when every part of the order is good:
packing and shipping.

Printers and cooling run 24/7. Every manual task needs a station AND an operator, and operators are
only available during shifts. Time unit: hours, t = 0 is Monday 00:00.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simpy
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "baseline.yaml"
SHIFT_START = (6, 14, 22)
MANUAL = ("prep", "depowder", "blast", "dye", "finish", "qc", "pack")


def load_config(path: str | Path | None = None, **overrides) -> dict:
    """Load the YAML config and apply dotted overrides, e.g. {"resources.printers": 4}."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node[p]
        if last not in node:
            raise KeyError(f"unknown parameter {key}")
        node[last] = value
    return cfg


def on_shift(t: float, shifts: int, days: int) -> bool:
    """True when operators are present at time t (hours from Monday 00:00)."""
    day, hour = int(t // 24), t % 24
    for s in range(shifts):
        start = SHIFT_START[s]
        end = start + 8
        if start <= hour < min(end, 24) and day % 7 < days:
            return True
        if end > 24 and hour < end - 24 and (day - 1) % 7 < days:  # night shift after midnight
            return True
    return False


@dataclass
class Part:
    pid: int
    order: "Order"
    volume: float
    dye: bool
    t_ready: float  # time it entered the nesting queue
    reprints: int = 0


@dataclass
class Order:
    oid: int
    t_arrival: float
    n_parts: int
    good: int = 0
    t_shipped: float | None = None


@dataclass
class Job:
    jid: int
    parts: list
    height_cm: float
    print_h: float


@dataclass
class Stats:
    busy: dict = field(default_factory=lambda: {})        # station -> processing hours in the window
    waits: dict = field(default_factory=lambda: {})       # station -> list of waiting times
    operator_busy: float = 0.0
    printer_state: dict = field(default_factory=lambda: {"printing": 0.0, "swap": 0.0})
    jobs: list = field(default_factory=list)              # (t_start_print, height, n_parts, fill)
    rejects: int = 0


class Workshop:
    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        r, sh = cfg["resources"], cfg["shifts"]
        self.t0 = cfg["warmup_weeks"] * 168.0
        self.t_end = cfg["horizon_weeks"] * 168.0     # end of arrivals and of the measurement window
        self.drain = cfg.get("drain_weeks", 2) * 168.0  # then the workshop finishes the orders in progress
        self.shifts, self.days = sh["count"], sh["days_per_week"]
        self.operators = simpy.PriorityResource(self.env, capacity=r["operators_per_shift"])
        self.printers = simpy.Resource(self.env, capacity=r["printers"])
        self.build_units = simpy.Resource(self.env, capacity=r["build_units"])
        self.cap = {"prep": r["operators_per_shift"], "depowder": r["depowder_stations"],
                    "blast": r["blast_cabinets"], "dye": r["dye_vats"], "finish": r["finishing_benches"],
                    "qc": r["qc_benches"], "pack": r["operators_per_shift"]}
        self.st = {k: simpy.Resource(self.env, capacity=c) for k, c in self.cap.items() if k not in ("dye",)}
        self.nest_q: list[Part] = []
        self.dye_q: list[tuple[float, Part]] = []
        self.nest_signal = self.env.event()
        self.dye_signal = self.env.event()
        self.orders: list[Order] = []
        self.stats = Stats()
        self._pid = self._jid = 0
        self.env.process(self._shift_calendar())
        self.env.process(self._arrivals())
        self.env.process(self._nesting())
        for _ in range(r["dye_vats"]):
            self.env.process(self._dye_vat())

    # ------------------------------------------------------------------ helpers
    def _u(self, pair):
        return float(self.rng.uniform(*pair))

    def _record(self, station, wait, busy):
        """Keep statistics only for activity that starts inside the measurement window."""
        if self.t0 <= self.env.now < self.t_end:
            self.stats.waits.setdefault(station, []).append(wait)
            self.stats.busy[station] = self.stats.busy.get(station, 0.0) + busy

    def _wake(self, name):
        ev = getattr(self, name)
        if not ev.triggered:
            ev.succeed()
        setattr(self, name, self.env.event())

    def manual(self, station, minutes):
        """Generator: take a station, then an operator, work, release both."""
        t_arr = self.env.now
        with self.st[station].request() as s:
            yield s
            with self.operators.request(priority=1) as op:
                yield op
                dur = minutes / 60.0
                self._record(station, self.env.now - t_arr, dur)
                if self.t0 <= self.env.now < self.t_end:
                    self.stats.operator_busy += dur
                yield self.env.timeout(dur)

    # ------------------------------------------------------------------ processes
    def _shift_calendar(self):
        """Operators are seized by a high-priority 'off shift' request outside shift hours."""
        n = self.cfg["resources"]["operators_per_shift"]
        held = []
        t = 0.0
        while t < self.t_end + self.drain + 24:
            want_off = not on_shift(t, self.shifts, self.days)
            if want_off and not held:
                held = [self.operators.request(priority=0) for _ in range(n)]
            elif not want_off and held:
                for req in held:
                    if req.triggered:
                        self.operators.release(req)
                    else:
                        req.cancel()
                held = []
            yield self.env.timeout(1.0)
            t += 1.0

    def _arrivals(self):
        d = self.cfg["demand"]
        while self.env.now < self.t_end:
            day, hour = int(self.env.now // 24), self.env.now % 24
            if day % 7 >= 5 or not 8 <= hour < 18:  # orders come in on working days, 08:00 to 18:00
                yield self.env.timeout(1.0 - (self.env.now % 1.0) or 1.0)
                continue
            rate = d["orders_per_day"] / 10.0  # per hour over the 10 h window
            yield self.env.timeout(float(self.rng.exponential(1.0 / rate)))
            if not (day % 7 < 5 and 8 <= self.env.now % 24 < 18) or self.env.now >= self.t_end:
                continue
            n = int(self.rng.integers(d["parts_per_order"][0], d["parts_per_order"][1] + 1))
            order = Order(len(self.orders), self.env.now, n)
            self.orders.append(order)
            vol = d["part_volume_cm3"]
            for _ in range(n):
                v = float(self.rng.lognormal(np.log(vol["median"]), vol["sigma"]))
                self.nest_q.append(Part(self._pid, order, min(v, 1500.0), bool(self.rng.random() < d["dye_share"]),
                                        self.env.now))
                self._pid += 1
            self._wake("nest_signal")

    def _nesting(self):
        """Build a job when the queue fills a build, or the oldest part has waited too long."""
        n_cfg, tm = self.cfg["nesting"], self.cfg["times"]
        cap = n_cfg["job_capacity_cm3"]
        while True:
            if not self.nest_q:
                yield self.nest_signal
                continue
            vol = sum(p.volume for p in self.nest_q)
            oldest = min(p.t_ready for p in self.nest_q)
            printer_idle = self.printers.count < self.printers.capacity
            late = self.env.now - oldest >= n_cfg["max_wait_h"]
            if vol >= cap or (late and (vol >= n_cfg["min_fill"] * cap or printer_idle)):
                yield from self.manual("prep", self._u(tm["prep_min"]))
                self.nest_q.sort(key=lambda p: p.t_ready)
                chosen, used = [], 0.0
                for p in list(self.nest_q):
                    if used + p.volume <= cap or not chosen:
                        chosen.append(p)
                        used += p.volume
                for p in chosen:
                    self.nest_q.remove(p)
                h = min(tm["max_height_cm"], 2.0 + used / (tm["footprint_cm2"] * tm["packing_density"]))
                job = Job(self._jid, chosen, h, tm["print_base_h"] + tm["print_h_per_cm"] * h)
                self._jid += 1
                self.env.process(self._job(job, used / cap))
            else:
                # re-check when new parts arrive or after one hour
                yield self.nest_signal | self.env.timeout(1.0)

    def _job(self, job: Job, fill: float):
        tm = self.cfg["times"]
        t_arr = self.env.now
        bu = self.build_units.request()
        yield bu
        with self.printers.request() as pr:
            yield pr
            if self.t0 <= self.env.now < self.t_end:
                self.stats.jobs.append((self.env.now, job.height_cm, len(job.parts), fill))
                self.stats.printer_state["swap"] += tm["swap_min"] / 60
                self.stats.printer_state["printing"] += job.print_h
            self._record("printer", self.env.now - t_arr, job.print_h + tm["swap_min"] / 60)
            yield self.env.timeout(tm["swap_min"] / 60 + job.print_h)
        yield self.env.timeout(tm["cooling_ratio"] * job.print_h)
        yield from self.manual("depowder", self._u(tm["depowder_min"]))
        self.build_units.release(bu)
        for p in job.parts:
            self.env.process(self._part(p))

    def _part(self, p: Part):
        tm = self.cfg["times"]
        yield from self.manual("blast", self._u(tm["blast_min_per_part"]))
        if p.dye:
            done = self.env.event()
            self.dye_q.append((self.env.now, p, done))
            self._wake("dye_signal")
            yield done
        yield from self.manual("finish", self._u(tm["finish_min_per_part"]))
        yield from self.manual("qc", self._u(tm["qc_min_per_part"]))
        if self.rng.random() < tm["reject_rate"]:
            if self.t0 <= self.env.now < self.t_end:
                self.stats.rejects += 1
            p.reprints += 1
            p.t_ready = self.env.now
            self.nest_q.append(p)
            self._wake("nest_signal")
            return
        p.order.good += 1
        if p.order.good == p.order.n_parts:
            self.env.process(self._ship(p.order))

    def _dye_vat(self):
        tm = self.cfg["times"]
        while True:
            if not self.dye_q:
                yield self.dye_signal
                continue
            oldest = min(t for t, _, _ in self.dye_q)
            if len(self.dye_q) >= tm["dye_capacity"] or self.env.now - oldest >= tm["dye_max_wait_h"]:
                batch = self.dye_q[:tm["dye_capacity"]]
                del self.dye_q[:tm["dye_capacity"]]
                t_arr = self.env.now
                with self.operators.request(priority=1) as op:
                    yield op
                    if self.t0 <= self.env.now < self.t_end:
                        self.stats.operator_busy += tm["dye_load_min"] / 60
                    self._record("dye", self.env.now - t_arr, tm["dye_cycle_min"] / 60)
                    yield self.env.timeout(tm["dye_load_min"] / 60)
                yield self.env.timeout(tm["dye_cycle_min"] / 60)
                for _, _, ev in batch:
                    ev.succeed()
            else:
                yield self.dye_signal | self.env.timeout(0.5)

    def _ship(self, order: Order):
        yield from self.manual("pack", self._u(self.cfg["times"]["pack_min_per_order"]))
        order.t_shipped = self.env.now

    # ------------------------------------------------------------------ run and KPIs
    def run(self) -> dict:
        self.env.run(until=self.t_end + self.drain)
        return self.kpis()

    def shift_hours(self) -> float:
        hours = np.arange(self.t0, self.t_end, 0.25) + 0.125
        return float(sum(on_shift(t, self.shifts, self.days) for t in hours) * 0.25)

    def kpis(self) -> dict:
        window_h = self.t_end - self.t0
        weeks = window_h / 168
        shift_h = self.shift_hours()
        window = [o for o in self.orders if o.t_arrival >= self.t0]  # arrivals stop at t_end
        done = [o for o in window if o.t_shipped is not None]
        lead = np.array([o.t_shipped - o.t_arrival for o in done]) if done else np.array([np.nan])
        late_orders = [o for o in window if o.t_shipped is None or o.t_shipped > self.t_end]
        unfinished = [o for o in window if o.t_shipped is None]
        # orders still open after the drain period count with their age then: a lower bound that keeps
        # saturated configurations comparable (equal to the plain mean when every order shipped)
        cens = [(o.t_shipped if o.t_shipped is not None else self.t_end + self.drain) - o.t_arrival for o in window]
        shipped_parts = sum(o.n_parts for o in self.orders if o.t_shipped and self.t0 <= o.t_shipped < self.t_end)
        util = {}
        for st, cap in self.cap.items():
            avail = (window_h if st == "dye" else shift_h) * cap
            util[st] = self.stats.busy.get(st, 0.0) / avail if avail else 0.0
        util["printer"] = self.stats.busy.get("printer", 0.0) / (window_h * self.printers.capacity)
        util["operators"] = self.stats.operator_busy / (shift_h * self.operators.capacity)
        waits = {k: float(np.mean(v)) for k, v in self.stats.waits.items()}
        jobs = self.stats.jobs
        return {
            "orders_shipped": len(done),
            "parts_shipped_per_week": shipped_parts / weeks,
            "lead_time_mean_h": float(np.mean(lead)) if done else float("nan"),
            "lead_time_censored_mean_h": float(np.mean(cens)) if cens else float("nan"),
            "lead_time_p50_h": float(np.percentile(lead, 50)) if done else float("nan"),
            "lead_time_p90_h": float(np.percentile(lead, 90)) if done else float("nan"),
            "orders_open_at_end": len(late_orders),
            "orders_unfinished_after_drain": len(unfinished),
            "parts_waiting_nesting_end": len(self.nest_q),
            "jobs_per_week": len(jobs) / weeks,
            "mean_build_fill": float(np.mean([j[3] for j in jobs])) if jobs else 0.0,
            "rejects": self.stats.rejects,
            "utilisation": util,
            "mean_wait_h": waits,
            "shift_hours_per_week": shift_h / weeks,
        }


def simulate(cfg: dict | None = None, seed: int = 0) -> dict:
    return Workshop(cfg or load_config(), seed).run()
