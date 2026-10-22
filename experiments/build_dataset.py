"""Simulate 400 random workshop configurations for the surrogate (about 5 min on 2 cores)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amtwin.surrogate import build_dataset  # noqa: E402

if __name__ == "__main__":
    t = time.time()
    df = build_dataset(400)
    df.to_csv(Path(__file__).resolve().parent.parent / "results" / "surrogate_dataset.csv", index=False)
    print(f"{len(df)} configurations in {time.time() - t:.0f} s")
