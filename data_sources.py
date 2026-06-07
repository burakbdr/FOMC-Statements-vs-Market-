"""
Data layer for the FOMC Communication dashboard.

This module produces three artifacts that the rest of the pipeline consumes.
Keeping the "data contract" explicit means the visualization never cares where
the numbers came from -- real scrape or sample -- which is what we want once
this goes on a website with a scheduled refresh.

Data contract (all written into ./data):
  - statements.csv : columns [date, word_count]      (one row per FOMC meeting)
  - market.csv     : columns [date, vix]             (daily series)
  - freqs.json     : {chair_name: {word: weight, ...}}  (word-cloud input)

Two backends:
  - REAL  : fetch_statements_real() / fetch_market_real()  -> live internet
  - SAMPLE: build_sample_*()  -> deterministic, offline, for demos & CI

Switch with build.py --sample / --real.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# Fed chairs (historical fact). Each meeting is attributed to the sitting chair
# by date. End date is exclusive; the final chair has end=None (open ended).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chair:
    name: str
    start: date
    end: date | None  # exclusive

    def covers(self, d: date) -> bool:
        return d >= self.start and (self.end is None or d < self.end)


CHAIRS: list[Chair] = [
    Chair("Greenspan", date(1987, 8, 11), date(2006, 2, 1)),
    Chair("Bernanke", date(2006, 2, 1), date(2014, 2, 3)),
    Chair("Yellen", date(2014, 2, 3), date(2018, 2, 5)),
    Chair("Powell", date(2018, 2, 5), date(2026, 5, 22)),
    Chair("Warsh", date(2026, 5, 22), None),  # sworn in 2026-05-22
]

# Single source of truth for accent colors, reused by the site.
CHAIR_COLORS = {
    "Greenspan": "#6B705C",
    "Bernanke": "#9C6644",
    "Yellen": "#3D5A80",
    "Powell": "#7D4F9E",
    "Warsh": "#B5161C",
}


def chair_for(d: date) -> str:
    for c in CHAIRS:
        if c.covers(d):
            return c.name
    return "Unknown"


# --------------------------------------------------------------------------- #
# REAL backend (runs on a machine with open internet -- not this container).
# Kept deliberately small and documented; refine the scraper as needed.
# --------------------------------------------------------------------------- #
def fetch_market_real(start: date, end: date) -> pd.DataFrame:
    """
    Daily VIX (^VIX).

    Yahoo/yfinance is tried first but is flaky (auth changes, rate limits, the
    'no timezone found' error). If it returns nothing we fall back to Stooq,
    which serves the same series as a plain CSV with no auth.
    """
    df = _fetch_market_yfinance(start, end)
    if df is None or df.empty:
        print("      yfinance returned no data — falling back to Stooq ...")
        df = _fetch_market_stooq(start, end)
    if df is None or df.empty:
        raise RuntimeError(
            "Could not fetch market data from yfinance or Stooq.\n"
            "  - yfinance is likely throttled/blocked by Yahoo on this network; "
            "upgrading often helps:  pip install -U yfinance\n"
            "  - Stooq rate-limits per IP per day — wait a bit and retry.\n"
            "  - Or build with sample data now:  python build.py --sample"
        )
    return df.sort_values("date").reset_index(drop=True)


def _naive_dates(s: "pd.Series") -> "pd.Series":
    """Normalize a date column to tz-naive midnight, regardless of input tz."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


def _fetch_market_yfinance(start: date, end: date) -> pd.DataFrame | None:
    """
    Download the VIX in one call. `multi_level_index=False` keeps the column index
    flat and `['Close']` returns the closing series mapped onto our contract.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        close = yf.download(
            ["^VIX"], start=start, end=end, multi_level_index=False
        )["Close"]
    except Exception:
        return None

    if close is None or close.empty:
        return None

    m = close.reset_index()
    m.columns = ["date", "vix"][: len(m.columns)]
    if "vix" not in m.columns:
        return None  # came back empty -> let the Stooq fallback try

    m["date"] = _naive_dates(m["date"])
    return m[["date", "vix"]].dropna().reset_index(drop=True)


def _fetch_market_stooq(start: date, end: date) -> pd.DataFrame:
    """
    Free, auth-free CSV from Stooq (^vix), long history.

    Stooq blocks unknown User-Agents and rate-limits per IP/day; in those cases it
    returns a short text/HTML message instead of CSV. We send a browser UA, verify
    the body really is CSV before parsing, skip malformed lines, and retry — and
    never let a bad response raise.
    """
    import io
    import time

    import requests

    UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

    def one(sym: str, name: str) -> pd.DataFrame | None:
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        for attempt in range(3):
            try:
                txt = requests.get(url, timeout=60, headers={"User-Agent": UA}).text
            except Exception:
                time.sleep(1.5)
                continue
            first = txt.strip().splitlines()[0].lower() if txt.strip() else ""
            if not first.startswith("date,"):
                # not CSV (block page / "Exceeded the daily hits limit" / "No data")
                time.sleep(1.5)
                continue
            try:
                df = pd.read_csv(io.StringIO(txt), on_bad_lines="skip")
            except Exception:
                return None
            if "Date" not in df.columns or "Close" not in df.columns:
                return None
            df = df.rename(columns={"Date": "date", "Close": name})[["date", name]]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df[name] = pd.to_numeric(df[name], errors="coerce")
            return df.dropna()
        return None

    vix = one("^vix", "vix")
    if vix is None:
        return pd.DataFrame(columns=["date", "vix"])
    m = vix[(vix["date"] >= pd.Timestamp(start)) & (vix["date"] <= pd.Timestamp(end))]
    return m[["date", "vix"]].reset_index(drop=True)


def fetch_statements_real(start_year: int = 1994) -> pd.DataFrame:
    """
    Scrape FOMC post-meeting statements and count words.

    The Fed has issued a statement after every meeting since 1994. Modern
    statements (2019+) live at:
        https://www.federalreserve.gov/newsevents/pressreleases/monetaryYYYYMMDDa.htm
    Older ones are linked from the historical FOMC calendar pages. This helper
    walks the calendar index pages, follows each "statement" link, strips HTML,
    and counts words. Network access to federalreserve.gov is required.
    """
    import re

    import requests
    from bs4 import BeautifulSoup

    sess = requests.Session()
    sess.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    def statement_text(html: str) -> str:
        """
        Return the STATEMENT BODY text only. The press-release text sits in
        <div id="article"> on modern federalreserve.gov pages; older layouts use
        #content / <main> / <article>. Reading the whole page (the previous bug)
        swept in menus, sidebars, "related materials" and the footer — which is how
        a ~300-word statement became 30,000+.
        """
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        node = (soup.find("div", id="article")
                or soup.find(id="content")
                or soup.find("main")
                or soup.find("article"))
        container = node if node is not None else (soup.body or soup)
        return re.sub(r"\s+", " ", container.get_text(" ")).strip()

    rows: list[dict] = []
    # Recent years use the rolling calendar; historical years use per-year pages.
    index_urls = [
        "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    ] + [
        f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
        for y in range(start_year, datetime.now().year)
    ]

    NON_HTML = (".pdf", ".csv", ".xls", ".xlsx", ".zip", ".doc", ".docx", ".xml")
    # Statement URL shapes across eras: modern .../monetaryYYYYMMDDa.htm and the old
    # boarddocs/press/{monetary,general}/YYYY/YYYYMMDD/ pages used 1994–2005.
    STMT_HREF = re.compile(
        r"(monetary\d{8}a?\d?|boarddocs/press/(?:monetary|general)/\d{4}/\d{8})", re.I)
    # Dated links that are NOT the policy statement (would pollute the set).
    NOT_STMT = re.compile(
        r"(minutes|beige|projtabl|transcript|longerrun|goals|pressconf|conference)", re.I)

    seen: set[str] = set()
    for idx_url in index_urls:
        try:
            html = sess.get(idx_url, timeout=30).text
        except Exception:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            label = a.get_text(" ").strip().lower()
            # Match by link label OR by URL shape: older historical pages label the
            # statement link by date rather than the word "statement".
            looks_stmt = ("statement" in label) or bool(STMT_HREF.search(href))
            if not looks_stmt or NOT_STMT.search(href):
                continue
            # Only follow the HTML statement page — never the meeting "material",
            # minutes, or transcript files (e.g. FOMC19980818material.pdf).
            if href.lower().split("?")[0].endswith(NON_HTML):
                continue
            m = re.search(r"(\d{8})", href)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y%m%d").date()
            url = href if href.startswith("http") else "https://www.federalreserve.gov" + href
            if url in seen:
                continue
            seen.add(url)
            try:
                resp = sess.get(url, timeout=30)
            except Exception:
                continue
            # Guard against PDFs/binaries served without a telltale extension.
            if "html" not in resp.headers.get("Content-Type", "").lower():
                print(f"      skipped {d}: not an HTML page ({url})")
                continue
            text = statement_text(resp.text)
            n = len(re.findall(r"[A-Za-z']+", text))
            # A genuine FOMC statement is ~150–1000 words. Anything outside a
            # generous band means we grabbed the wrong thing (index page, full
            # page chrome, etc.) — skip it rather than poison the average.
            if not (20 <= n <= 2000):
                print(f"      skipped {d}: implausible word count ({n}) from {url}")
                continue
            rows.append({"date": d, "word_count": n, "text": text})

    if not rows:
        return pd.DataFrame(columns=["date", "word_count", "text"])
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# SAMPLE backend -- deterministic, offline. Reproduces the *shape* of the real
# story (Apollo / Torsten Slok): statement length crept up over time, peaked in
# the Yellen years, eased under Powell, and -- the news hook -- is expected to
# fall back toward Greenspan-era brevity under Warsh. Numbers are illustrative.
# --------------------------------------------------------------------------- #
def _fomc_meeting_dates(start_year: int, end: date) -> list[date]:
    """~8 scheduled meetings/year. Approximate but evenly spread per year."""
    months = [1, 3, 5, 6, 7, 9, 11, 12]
    days = {1: 31, 3: 18, 5: 1, 6: 17, 7: 30, 9: 17, 11: 5, 12: 17}
    out: list[date] = []
    for y in range(start_year, end.year + 1):
        for m in months:
            d = date(y, m, days[m])
            if d <= end:
                out.append(d)
    return out


def build_sample_statements(end: date, market: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Word count per meeting following the documented historical arc. Only real,
    past meetings are generated — no projected/future point. A new chair (e.g.
    Warsh) shows up on his own once a real statement exists for him.
    """
    rng = random.Random(seed)

    # (year, baseline word count) anchors; interpolated between, + noise.
    anchors = [
        (1994, 95), (1999, 150), (2003, 210), (2006, 260),
        (2008, 480), (2010, 700), (2013, 820), (2015, 880),
        (2017, 560), (2019, 470), (2020, 540), (2022, 510),
        (2024, 480), (2026, 470),
    ]

    def baseline(y: float) -> float:
        if y <= anchors[0][0]:
            return anchors[0][1]
        if y >= anchors[-1][0]:
            return anchors[-1][1]
        for (y0, v0), (y1, v1) in zip(anchors, anchors[1:]):
            if y0 <= y <= y1:
                t = (y - y0) / (y1 - y0)
                return v0 + t * (v1 - v0)
        return anchors[-1][1]

    # Apollo thesis, made explicit in the sample: higher pre-meeting volatility
    # -> longer statements. We compute each meeting's 6-week-prior VIX from the
    # market series and add a sensitivity term on top of the trend.
    VIX_SENS = 6.5          # words added per VIX point above the calm baseline
    VIX_CALM = 17.0

    _m = market.copy()
    _m["date"] = pd.to_datetime(_m["date"])
    mkt = _m.set_index("date")["vix"].sort_index()

    def prior_vix(d: date) -> float:
        lo = pd.Timestamp(d) - pd.Timedelta(days=42)
        w = mkt.loc[lo: pd.Timestamp(d)]
        return float(w.mean()) if len(w) else VIX_CALM

    rows = []
    for d in _fomc_meeting_dates(1994, end):
        yfrac = d.year + (d.month - 1) / 12
        vix_term = VIX_SENS * max(0.0, prior_vix(d) - VIX_CALM)
        base = baseline(yfrac) + vix_term
        wc = max(40, int(base + rng.gauss(0, base * 0.05)))
        rows.append({"date": d, "word_count": wc})
    return pd.DataFrame(rows)


def build_sample_market(end: date, seed: int = 11) -> pd.DataFrame:
    """Daily VIX with realistic regime shifts and volatility spikes."""
    rng = random.Random(seed)
    start = date(1993, 11, 1)  # a little before 1994 so 6wk windows are covered
    n_days = (end - start).days

    # Volatility events: (start, end, peak VIX).
    vol_events = [
        (date(1998, 8, 1), date(1998, 11, 1), 45),
        (date(2001, 9, 1), date(2001, 11, 1), 43),
        (date(2002, 7, 1), date(2002, 10, 15), 45),
        (date(2008, 9, 1), date(2009, 4, 1), 80),
        (date(2010, 5, 1), date(2010, 7, 1), 45),
        (date(2011, 8, 1), date(2011, 11, 1), 48),
        (date(2015, 8, 1), date(2015, 10, 1), 40),
        (date(2018, 12, 1), date(2019, 1, 15), 36),
        (date(2020, 2, 20), date(2020, 6, 1), 82),
        (date(2022, 1, 1), date(2022, 11, 1), 35),
        (date(2025, 4, 1), date(2025, 6, 1), 38),
    ]

    def vol_bump(d: date) -> float:
        bump = 0.0
        for s, e, peak in vol_events:
            if s <= d <= e:
                span = (e - s).days
                pos = (d - s).days / span
                shape = math.sin(math.pi * pos) ** 0.6  # rise then fade
                bump = max(bump, (peak - 16) * shape)
        return bump

    rows = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:  # skip weekends
            continue
        vix = 13 + 5 * abs(rng.gauss(0, 1)) + vol_bump(d)
        rows.append({"date": d, "vix": round(vix, 2)})
    return pd.DataFrame(rows)


def build_sample_freqs() -> dict[str, dict[str, float]]:
    """
    Representative term weights per chair, capturing each era's signature
    language. Replace with real corpus frequencies once statements are scraped
    (process.compute_freqs_from_corpus does that when raw text is available).
    """
    return {
        "Greenspan": {
            "measured": 95, "accommodation": 70, "productivity": 66, "pace": 60,
            "considerable": 58, "sustainable": 50, "pressures": 46, "balanced": 44,
            "policy": 40, "growth": 38, "spending": 34, "modest": 30, "removed": 28,
            "outlook": 26, "household": 24, "irrational": 18,
        },
        "Bernanke": {
            "securities": 92, "purchases": 84, "accommodative": 78, "mortgage": 70,
            "substantial": 64, "asset": 60, "downside": 52, "highly": 48,
            "agency": 44, "longer-term": 40, "recovery": 38, "subdued": 34,
            "improvement": 30, "exceptionally": 26, "strains": 22,
        },
        "Yellen": {
            "gradual": 90, "normalization": 80, "labor": 76, "transitory": 66,
            "appropriate": 60, "balance": 56, "reinvesting": 50, "patient": 46,
            "soften": 40, "median": 36, "running": 32, "symmetric": 28,
            "longer-run": 26, "firming": 22,
        },
        "Powell": {
            "inflation": 96, "employment": 82, "pandemic": 70, "maximum": 64,
            "transitory": 58, "supply": 54, "accommodative": 50, "uncertain": 46,
            "highly": 40, "restrictive": 38, "resilient": 32, "elevated": 30,
            "appropriate": 26, "patient": 22,
        },
        "Warsh": {
            "price": 88, "stability": 80, "discipline": 64, "focus": 58,
            "restraint": 52, "clarity": 48, "credibility": 40, "simple": 36,
            "mandate": 30, "anchored": 26, "decisive": 22, "brief": 18,
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def write_dataset(use_sample: bool, end: date | None = None) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    end = end or date.today()

    if use_sample:
        market = build_sample_market(end)
        statements = build_sample_statements(end, market)
        freqs = build_sample_freqs()
        source = "sample"
    else:
        statements = fetch_statements_real()
        market = fetch_market_real(date(1993, 11, 1), end)
        if statements.empty:
            raise RuntimeError(
                "No FOMC statements were scraped. Check network access to "
                "federalreserve.gov, or the calendar page layout may have changed."
            )
        # REAL word-cloud frequencies, computed from the actual statement text —
        # never the representative sample weights. A chair with no scraped
        # statements simply won't appear (e.g. Warsh before his first statement).
        from process import compute_freqs_from_corpus  # deferred: avoids import cycle
        texts_by_chair: dict[str, list[str]] = {}
        for _, r in statements.iterrows():
            ch = chair_for(r["date"])
            texts_by_chair.setdefault(ch, []).append(r["text"])
        freqs = compute_freqs_from_corpus(texts_by_chair)
        statements = statements.drop(columns=["text"])  # don't persist full corpus
        source = "real"

    statements.to_csv(DATA_DIR / "statements.csv", index=False)
    market.to_csv(DATA_DIR / "market.csv", index=False)
    (DATA_DIR / "freqs.json").write_text(json.dumps(freqs, indent=2))
    (DATA_DIR / "meta.json").write_text(
        json.dumps({"source": source, "generated": datetime.now().isoformat()}, indent=2)
    )
    return {"source": source, "n_statements": len(statements), "n_market_days": len(market)}
