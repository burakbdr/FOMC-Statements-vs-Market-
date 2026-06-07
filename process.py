"""
Processing layer.

Turns the raw data-contract files into the single table the charts need:
one row per FOMC meeting with
  - word_count       (statement length)
  - chair            (sitting chair at that date)
  - vix_prior_6w     (MAX VIX over the 6 weeks BEFORE the meeting -- the Apollo
                      thesis is that pre-meeting volatility drives wordiness)

Keeping this separate from fetching and rendering means the website refresh is
just: data_sources.write_dataset() -> process.build_meeting_table() -> render.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from data_sources import DATA_DIR, chair_for

PRIOR_WINDOW_DAYS = 42  # 6 weeks


def build_meeting_table() -> pd.DataFrame:
    statements = pd.read_csv(DATA_DIR / "statements.csv", parse_dates=["date"])
    market = pd.read_csv(DATA_DIR / "market.csv", parse_dates=["date"]).sort_values("date")

    statements = statements.sort_values("date").reset_index(drop=True)
    market = market.set_index("date")

    vix_prior = []
    for d in statements["date"]:
        window = market.loc[d - pd.Timedelta(days=PRIOR_WINDOW_DAYS): d]
        # Apollo uses the MAX VIX in the prior 6 weeks (peak pre-meeting stress).
        vix_prior.append(round(window["vix"].max(), 2) if len(window) else None)

    statements["vix_prior_6w"] = vix_prior  # = max VIX in the 6 weeks before
    statements["chair"] = statements["date"].dt.date.map(chair_for)
    return statements.dropna(subset=["vix_prior_6w"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real word-cloud frequencies from raw statement text (used by the real
# pipeline once statement bodies are saved alongside their dates).
# --------------------------------------------------------------------------- #
_STOPWORDS = set(
    """a an the and or but if then than that this these those of to in on for with as at by from is are was
    were be been being it its has have had will would shall should may might can could do does did not no
    so such we our committee federal reserve open market policy rate rates percent point points meeting
    monetary economic economy over per into about which while when also more most other their there they
    been further continue continues remain remains expects expected expect range target year years month
    months term recent data outlook conditions including based seek seeks two one set out up down off
    """.split()
)


def compute_freqs_from_corpus(texts_by_chair: dict[str, list[str]], top_n: int = 40) -> dict:
    """Tokenize, drop stopwords, return top_n weighted terms per chair."""
    out: dict[str, dict[str, float]] = {}
    for chair, texts in texts_by_chair.items():
        counter: Counter = Counter()
        for t in texts:
            for w in re.findall(r"[a-z][a-z'\-]{2,}", t.lower()):
                if w not in _STOPWORDS:
                    counter[w] += 1
        out[chair] = dict(counter.most_common(top_n))
    return out
