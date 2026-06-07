# The Length of Fed Words — FOMC Communication Monitor

An interactive dashboard exploring a simple idea (popularized by Apollo's Torsten
Sløk): **when markets get volatile, the Fed's post-meeting statement gets longer.**
With Kevin Warsh sworn in as Fed chair (May 2026) on a promise to simplify Fed
communication, statement length may fall back toward Greenspan-era brevity — the
news hook this project was built around.

The screen has these parts:

1. **Word count over time** — statement length per FOMC meeting (bars, colored by the
   sitting chair); the era story at a glance.
2. **Statements vs. the VIX** (scatter) — each meeting plotted against the max VIX in
   the six weeks before it, with raw and detrended (first-difference) correlations.
3. **Statements and the VIX over time** (Apollo-style) — word count and Max VIX as two
   lines on a dual axis; shows *where* they co-move (the crisis spikes).
4. **Fed funds rate vs. the 2-year Treasury** — the policy rate against the market's
   implied rate path (2Y above the funds rate = hikes priced, below = cuts). FRED data;
   appears only when `rates.csv` is present.
5. **The tone of the latest statement** — hawkish/dovish/neutral breakdown of the most
   recent statement (real mode only; never fabricated — omitted if it can't be computed).
6. **A word-cloud explorer** — click a chair to see the vocabulary that defined their
   statements.

A new chair appears automatically once a real statement exists for them — nothing is
projected or fabricated.

Built with **Plotly** (charts) and **wordcloud2.js** (clouds). The output is a single
self-contained `docs/index.html` — no server required.

---

## Quick start

```bash
pip install -r requirements.txt

python build.py            # live data (real) — default
python build.py --sample   # offline sample data (local dev only)
```

Open `docs/index.html` in any browser.

---

## How it's wired (and why)

The pipeline is split into three stages so the visualization never cares where the
numbers came from — which matters once this runs on a schedule for a website.

```
build.py  ──>  src/data_sources.py  ──>  data/*.csv, data/freqs.json
                       │                         │
                       │                         ▼
                       └──────────>  src/process.py  (meeting table:
                                          word_count, chair,
                                          vix_prior_6w)
                                                  │
                                                  ▼
                                     src/build_site.py  ──>  docs/index.html
```

**Data contract** (everything downstream reads only these):

| file             | columns / shape                                  |
|------------------|--------------------------------------------------|
| `statements.csv` | `date, word_count` (real mode also keeps `text`)        |
| `market.csv`     | `date, vix` (daily)                              |
| `freqs.json`     | `{chair: {word: weight, ...}}`                   |

Swap the backend (sample ↔ real) and nothing else changes.

### Sample vs. real

- **Sample** (`--sample`, default): deterministic, offline. It reproduces the *shape*
  of the real story — length crept up over time, spiked in crises, and tracks
  pre-meeting VIX — so the dashboard is fully functional with no network. Numbers are
  illustrative, not actual statement counts.
- **Real** (default): `fetch_market_real` pulls `^VIX` daily closes. It
  tries `yfinance` first and, because Yahoo is flaky, automatically falls back to
  **Stooq** (free, auth-free CSV) if yfinance returns nothing. `fetch_statements_real`
  scrapes the statement body from federalreserve.gov, counts words, **and keeps the
  text** so the per-chair word clouds are computed from the real corpus
  (`compute_freqs_from_corpus`). `fetch_rates_real` pulls the Fed funds rate (`DFF`)
  and 2-year yield (`DGS2`) from **FRED** — set `FRED_API_KEY` in the environment or a
  `.env` file at the project root. Without the key, the Fed-funds-vs-2Y section is just
  omitted (never faked).

  **Authenticity:** nothing is fabricated in real mode. A chair with no published
  statements (e.g. Warsh before his first one) simply doesn't appear — no projected
  bars, no representative word lists. If a market series, statement, or the rates/FRED
  or sentiment data isn't there, that section isn't shown. The `--sample` backend is
  illustrative only, for offline development.

  > If yfinance keeps failing, `pip install -U yfinance` often helps — but the Stooq
  > fallback means the build still works.

---

## Publishing to the web (later)

Because the build emits one static file, deployment is trivial:

- **GitHub Pages / Netlify / Cloudflare Pages:** publish the `docs/` folder (GitHub Pages: main branch, /docs).
- **Scheduled refresh:** run `python build.py --real` in a GitHub Action (cron),
  commit the regenerated `docs/index.html`.

No backend or database needed for the current scope.

---

## Notes & honesty

- The headline correlation reflects raw word count vs. prior VIX; the decades-long
  upward trend in statement length dilutes it, so the relationship reads most clearly
  in the crisis spikes (2008, 2020) on the Apollo-style chart, and in the scatter's
  detrended Δr.

Inspired by Apollo / Torsten Sløk.
