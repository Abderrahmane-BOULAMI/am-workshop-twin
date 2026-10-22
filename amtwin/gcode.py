"""Read an FDM G-code file: estimated print time, layers, height and filament, to feed the twin.

Time model per move: trapezoidal speed profile with constant acceleration (start and end at rest),
which is what most firmware planners do in the worst case. Slicer estimates in the header
(';TIME:' Cura, '; estimated printing time' PrusaSlicer) are read too, for comparison.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

WORD = re.compile(r"([A-Z])([-+]?\d*\.?\d+)")


@dataclass
class GcodeSummary:
    moves: int
    extrusion_moves: int
    layers: int
    max_z_mm: float
    filament_mm: float
    est_time_h: float
    slicer_time_h: float | None

    def as_dict(self):
        return self.__dict__.copy()


def move_time(length_mm: float, feed_mm_min: float, accel_mm_s2: float) -> float:
    """Seconds to travel length_mm starting and ending at rest, cruise speed feed, constant accel."""
    if length_mm <= 0:
        return 0.0
    v = feed_mm_min / 60.0
    if accel_mm_s2 <= 0:
        return length_mm / v
    if length_mm >= v * v / accel_mm_s2:  # reaches cruise speed
        return length_mm / v + v / accel_mm_s2
    return 2.0 * math.sqrt(length_mm / accel_mm_s2)  # triangle profile


def _slicer_time(line: str) -> float | None:
    if line.startswith(";TIME:"):
        return float(line.split(":")[1]) / 3600
    m = re.search(r"estimated printing time.*?=\s*(.*)", line)
    if m:
        total = 0
        for val, unit in re.findall(r"(\d+)\s*([dhms])", m.group(1)):
            total += int(val) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
        return total / 3600
    return None


def parse(text: str, accel_mm_s2: float = 1000.0) -> GcodeSummary:
    x = y = z = e = 0.0
    feed = 1500.0
    absolute, abs_e = True, True
    moves = ext_moves = 0
    filament = seconds = 0.0
    layer_z: set[float] = set()
    slicer = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(";"):
            slicer = slicer if slicer is not None else _slicer_time(line)
            continue
        code = line.split(";")[0].upper()
        if not code:
            continue
        words = dict(WORD.findall(code))
        if "G" in words:
            g = int(float(words["G"]))
            if g == 90:
                absolute = abs_e = True
            elif g == 91:
                absolute = abs_e = False
            elif g == 92:
                e = float(words.get("E", e))
            elif g == 28:
                x = y = z = 0.0
            elif g == 4:  # dwell: P ms or S s
                seconds += float(words.get("P", 0)) / 1000 + float(words.get("S", 0))
            elif g in (0, 1):
                nx = float(words.get("X", x if absolute else 0)) + (0 if absolute else x)
                ny = float(words.get("Y", y if absolute else 0)) + (0 if absolute else y)
                nz = float(words.get("Z", z if absolute else 0)) + (0 if absolute else z)
                ne = float(words.get("E", e if abs_e else 0)) + (0 if abs_e else e)
                feed = float(words.get("F", feed))
                dist = math.dist((x, y, z), (nx, ny, nz))
                de = ne - e
                if dist > 0 or de != 0:
                    moves += 1
                    seconds += move_time(dist if dist > 0 else abs(de), feed, accel_mm_s2)
                if de > 0 and dist > 0:
                    ext_moves += 1
                    filament += de
                    layer_z.add(round(nz, 3))
                x, y, z, e = nx, ny, nz, ne
        elif "M" in words:
            m = int(float(words["M"]))
            if m == 82:
                abs_e = True
            elif m == 83:
                abs_e = False
    return GcodeSummary(moves, ext_moves, len(layer_z), max(layer_z) if layer_z else 0.0, filament,
                        seconds / 3600, slicer)


def parse_file(path: str | Path, accel_mm_s2: float = 1000.0) -> GcodeSummary:
    return parse(Path(path).read_text(encoding="utf-8", errors="ignore"), accel_mm_s2)


def make_box_gcode(size_mm: float = 40.0, height_mm: float = 10.0, layer_mm: float = 0.2,
                   feed: float = 3000.0) -> str:
    """Square perimeter box, one loop per layer: a test part whose time and filament are known."""
    lines = ["G90", "M82", "G28", "G92 E0"]
    e = 0.0
    n = int(round(height_mm / layer_mm))
    for i in range(1, n + 1):
        zc = round(i * layer_mm, 3)
        lines.append(f"G0 X0 Y0 Z{zc} F6000")
        for px, py in ((size_mm, 0), (size_mm, size_mm), (0, size_mm), (0, 0)):
            e += size_mm * 0.05
            lines.append(f"G1 X{px} Y{py} E{e:.4f} F{feed}")
    return "\n".join(lines) + "\n"
