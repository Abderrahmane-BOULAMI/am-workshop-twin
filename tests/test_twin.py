import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from amtwin.connectors import PrinterPoller, calibrate_print_time
from amtwin.gcode import make_box_gcode, move_time, parse
from amtwin.model import Workshop, apply_overrides, load_config, on_shift, simulate

SHORT = {"horizon_weeks": 4, "warmup_weeks": 1}


def cfg(**ov):
    return apply_overrides(load_config(), {**SHORT, **ov})


def test_shift_calendar():
    assert not on_shift(3, 2, 5)            # Monday 03:00, two shifts
    assert on_shift(6.5, 2, 5) and on_shift(21.9, 2, 5)
    assert not on_shift(22.1, 2, 5)
    assert on_shift(22.1, 3, 5) and on_shift(24 + 3, 3, 5)  # night shift crosses midnight
    assert not on_shift(5 * 24 + 10, 2, 5)  # Saturday
    assert on_shift(5 * 24 + 10, 2, 6)


def test_same_seed_same_result_and_seeds_differ():
    a, b, c = simulate(cfg(), 1), simulate(cfg(), 1), simulate(cfg(), 2)
    assert a == b
    assert a["lead_time_mean_h"] != c["lead_time_mean_h"]


def test_part_conservation():
    w = Workshop(cfg(), 3)
    w.run()
    created = sum(o.n_parts for o in w.orders)
    good = sum(o.good for o in w.orders)
    assert good <= created
    # every created part is either good, waiting for nesting, or somewhere in the flow
    assert len(w.nest_q) <= created - good


def test_lead_time_lower_bound():
    """No order can ship faster than print + cooling of its build."""
    c = cfg()
    k = simulate(c, 4)
    t = c["times"]
    min_print = t["print_base_h"] + t["print_h_per_cm"] * 2.0
    assert k["lead_time_p50_h"] > min_print * (1 + t["cooling_ratio"])


def test_utilisation_bounds():
    k = simulate(cfg(), 5)
    for name, u in k["utilisation"].items():
        assert 0.0 <= u <= 1.0 + 1e-9, name


def test_more_operators_never_hurt_much():
    base = np.mean([simulate(cfg(), s)["lead_time_mean_h"] for s in range(3)])
    more = np.mean([simulate(cfg(**{"resources.operators_per_shift": 5}), s)["lead_time_mean_h"] for s in range(3)])
    assert more < base


def test_no_operators_on_weekend_means_no_manual_work():
    c = cfg(**{"shifts.days_per_week": 5})
    w = Workshop(c, 6)
    w.env.run(until=5 * 24 + 1)          # Saturday 01:00
    before = w.stats.operator_busy
    w.env.run(until=7 * 24 - 1)          # Sunday 23:00
    # nothing started during the weekend (only work started before Friday 22:00 can finish)
    assert w.stats.operator_busy == pytest.approx(before)


def test_unknown_override_rejected():
    with pytest.raises(KeyError):
        apply_overrides(load_config(), {"resources.robots": 2})


def test_move_time_profiles():
    assert move_time(100, 6000, 0) == pytest.approx(1.0)
    long = move_time(1000, 6000, 1000)    # cruise at 100 mm/s
    assert long == pytest.approx(10 + 0.1)
    short = move_time(1, 6000, 1000)      # never reaches cruise
    assert short == pytest.approx(2 * np.sqrt(1 / 1000))


def test_gcode_box():
    g = make_box_gcode(size_mm=40, height_mm=10, layer_mm=0.2, feed=3000)
    s = parse(g, accel_mm_s2=0)
    assert s.layers == 50
    assert s.max_z_mm == pytest.approx(10.0)
    assert s.filament_mm == pytest.approx(50 * 4 * 40 * 0.05)
    # 50 layers x 160 mm at 50 mm/s + travel moves back to the corner (0 length except the z hop)
    assert s.est_time_h * 3600 == pytest.approx(50 * 160 / 50 + 50 * 0.2 / 100, rel=1e-6)


def test_gcode_slicer_header_and_relative_extrusion():
    g = "; estimated printing time (normal mode) = 1h 2m 3s\nM83\nG1 X10 E1 F600\nG1 X20 E1\n"
    s = parse(g, 0)
    assert s.slicer_time_h == pytest.approx((3600 + 120 + 3) / 3600)
    assert s.filament_mm == pytest.approx(2.0)


class FakeOctoPrint(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get("X-Api-Key") != "k":
            self.send_response(403); self.end_headers(); return
        if self.path == "/api/job":
            body = {"state": "Printing", "job": {"file": {"name": "bracket.gcode"}},
                    "progress": {"completion": 42.0, "printTimeLeft": 7200}}
        else:
            body = {"state": {"flags": {"printing": True}}}
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_poller_against_fake_server():
    srv = HTTPServer(("127.0.0.1", 0), FakeOctoPrint)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    st = PrinterPoller({"P1": (url, "k"), "P2": (url, "wrong"), "P3": ("http://127.0.0.1:9", "k")}).poll()
    srv.shutdown()
    assert st[0].state == "printing" and st[0].progress == pytest.approx(0.42)
    assert st[0].time_left_h == pytest.approx(2.0) and st[0].job == "bracket.gcode"
    assert st[1].state == "offline" and st[2].state == "offline"


def test_calibration_recovers_law():
    rng = np.random.default_rng(0)
    h = rng.uniform(5, 38, 40)
    y = 1.5 + 0.45 * h + rng.normal(0, 0.3, 40)
    fit = calibrate_print_time(h, y)
    assert fit["print_base_h"] == pytest.approx(1.5, abs=0.4)
    assert fit["print_h_per_cm"] == pytest.approx(0.45, abs=0.02)
