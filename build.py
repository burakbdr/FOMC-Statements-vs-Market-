"""
FOMC Communication dashboard — build pipeline.

Usage:
    python build.py            # sample data (offline, deterministic)
    python build.py --sample   # same as above
    python build.py --real     # live fetch (needs internet + yfinance, bs4)

Output: docs/index.html  (self-contained, deployable to any static host)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_sources import write_dataset          # noqa: E402
from build_site import render                    # noqa: E402
from process import build_meeting_table          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the FOMC communication dashboard.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="fetch live data (default)")
    mode.add_argument("--sample", action="store_true", help="offline sample data (dev only)")
    args = ap.parse_args()

    use_sample = args.sample  # real is the default now
    print(f"[1/3] Building dataset ({'sample' if use_sample else 'real'}) ...")
    try:
        info = write_dataset(use_sample=use_sample)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print(f"      {info}")

    print("[2/3] Processing meeting table ...")
    df = build_meeting_table()
    if df.empty:
        print("\nERROR: no meetings to plot. The market series came back empty, so "
              "the 6-week-prior VIX could not be computed for any meeting.\n"
              "Fix the market fetch (see README), or run:  python build.py --sample")
        sys.exit(1)
    print(f"      {len(df)} meetings, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    print("[3/3] Rendering site ...")
    out = render(df)
    print(f"      done → {out}")


if __name__ == "__main__":
    main()
