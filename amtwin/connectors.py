"""Machine connection: poll printers over HTTP and calibrate the twin from real build records.

`PrinterPoller` reads an OctoPrint-compatible REST API (GET /api/job and /api/printer, X-Api-Key
header). Industrial powder-bed printers expose similar job data through their own APIs; only the
`normalise` function changes. Completed builds are logged and `calibrate_print_time` refits the
twin's print-time law (hours = base + slope x build height) from them.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

import numpy as np


@dataclass
class PrinterStatus:
    printer: str
    state: str            # printing | idle | paused | error | offline
    job: str | None
    progress: float       # 0..1
    time_left_h: float | None
    t: float


def normalise(name: str, job: dict, printer: dict | None) -> PrinterStatus:
    flags = ((printer or {}).get("state") or {}).get("flags", {})
    text = (job.get("state") or "").lower()
    if flags.get("error") or "error" in text:
        state = "error"
    elif flags.get("paused") or "paus" in text:
        state = "paused"
    elif flags.get("printing") or text.startswith("printing"):
        state = "printing"
    elif text in ("offline", "closed"):
        state = "offline"
    else:
        state = "idle"
    prog = job.get("progress") or {}
    left = prog.get("printTimeLeft")
    return PrinterStatus(name, state, ((job.get("job") or {}).get("file") or {}).get("name"),
                         float(prog.get("completion") or 0) / 100, None if left is None else left / 3600, time.time())


class PrinterPoller:
    def __init__(self, printers: dict[str, tuple[str, str]], timeout: float = 3.0):
        """printers: name -> (base URL, API key)."""
        self.printers, self.timeout = printers, timeout

    def _get(self, url: str, key: str) -> dict:
        req = urllib.request.Request(url, headers={"X-Api-Key": key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def poll(self) -> list[PrinterStatus]:
        out = []
        for name, (base, key) in self.printers.items():
            try:
                job = self._get(base.rstrip("/") + "/api/job", key)
                try:
                    prn = self._get(base.rstrip("/") + "/api/printer", key)
                except Exception:  # /api/printer answers 409 when the printer is not connected
                    prn = None
                out.append(normalise(name, job, prn))
            except Exception:
                out.append(PrinterStatus(name, "offline", None, 0.0, None, time.time()))
        return out


def calibrate_print_time(heights_cm, hours) -> dict:
    """Least-squares fit of hours = base + slope x height, with the residual spread."""
    h, y = np.asarray(heights_cm, float), np.asarray(hours, float)
    A = np.vstack([np.ones_like(h), h]).T
    (base, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (base + slope * h)
    return {"print_base_h": float(base), "print_h_per_cm": float(slope),
            "residual_sd_h": float(resid.std(ddof=2)) if len(h) > 2 else 0.0, "n": int(len(h))}
