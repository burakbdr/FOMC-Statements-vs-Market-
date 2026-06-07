# The Length of Fed Words: FOMC Communication Monitor

An interactive dashboard built around a question popularized by Apollo's Torsten
Sløk: does the Fed write more when markets get rough? The data here suggests the
honest answer is "not really." Statement length tracks each chair's era and style far
more than it tracks volatility. The co-movement Apollo highlighted shows up mainly in
the crisis spikes (2008, 2020), not as a steady rule. Kevin Warsh became Fed chair in
May 2026; like every chair, he appears on the chart only once a real statement exists
for him.

The screen has these parts, in order:

1. **Word count over time:** statement length per FOMC meeting (bars, colored by the
   sitting chair); the era story at a glance.
2. **Statements vs. the VIX** (scatter): each meeting plotted against the peak VIX in
   the six weeks before it, with raw and detrended (first difference) correlations.
3. **Statements and the VIX over time** (Apollo style): word count and peak VIX as two
   lines on a dual axis; shows where they co-move (the crisis spikes).
4. **Fed funds rate vs. the two year Treasury:** the policy rate against the market's
   implied rate path (2Y above the funds rate means hikes priced, below means cuts).
   FRED data; appears only when `rates.csv` is present.
5. **A word cloud explorer:** click a chair to see the vocabulary that defined their
   statements.
6. **The tone of the latest statement:** a hawkish, dovish, or neutral reading of the
   most recent statement (real mode only; never fabricated, and omitted if it can't be
   computed). Shown last because it reads one statement, not the whole history.

A new chair appears automatically once a real statement exists for them; nothing is
projected or fabricated.

Built with **Plotly** (charts) and **wordcloud2.js** (clouds). The output is a single
static `docs/index.html`. It loads Plotly, wordcloud2, and fonts from their CDNs
(pinned with subresource integrity hashes), so viewing it needs an internet connection,
but no backend or server of your own is required.

---

## Quick start

```bash
pip install -r requirements.txt

python build.py            # live data (real); this is the default
python build.py --sample   # offline sample data (local dev only)
```

Then open the result. The simplest way, and the one that matches how GitHub Pages
serves it, is a local static server:

```bash
python -m http.server 8000
# then visit http://localhost:8000/docs/index.html
```

You can also just open `docs/index.html` directly in a browser.

---

## How it's wired (and why)

The pipeline is split into three stages so the visualization never cares where the
numbers came from. That separation matters once this runs on a schedule for a website.

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

| file             | columns / shape                                       |
|------------------|-------------------------------------------------------|
| `statements.csv` | `date, word_count` (statement text is not persisted)  |
| `market.csv`     | `date, vix` (daily)                                   |
| `rates.csv`      | `date, fed_funds, dgs2` (monthly; real mode only)     |
| `freqs.json`     | `{chair: {word: weight, ...}}`                        |
| `sentiment.json` | latest statement tone (real mode only; absent if off) |

Swap the backend (sample or real) and nothing else changes.

### Sample vs. real

- **Sample** (`--sample`): deterministic, offline. It reproduces the *shape* of the
  real story (length crept up over time, spiked in crises, and moves with pre-meeting
  VIX), so the dashboard is fully functional with no network. Numbers are illustrative,
  not actual statement counts.
- **Real** (the default): `fetch_market_real` pulls `^VIX` daily closes. It tries
  `yfinance` first and, because Yahoo is flaky, automatically falls back to **Stooq**
  (free, auth free CSV) if yfinance returns nothing. `fetch_statements_real` scrapes the
  statement body from federalreserve.gov and counts words. It uses the statement text in
  memory to build the per chair word clouds (`compute_freqs_from_corpus`), but only
  `date, word_count` is written to `statements.csv`; the raw text is not persisted.
  `fetch_rates_real` pulls the Fed funds rate (`DFF`) and two year yield (`DGS2`) from
  **FRED**: set `FRED_API_KEY` in the environment or a `.env` file at the project root.
  Without the key, the Fed funds vs. 2Y section is simply omitted (never faked).

  **Authenticity:** nothing is fabricated in real mode. A chair with no published
  statements (e.g. Warsh before his first one) simply doesn't appear: no projected bars,
  no representative word lists. If a market series, a statement, the FRED rates, or the
  sentiment data isn't there, that section isn't shown. The `--sample` backend is
  illustrative only, for offline development.

  > If yfinance keeps failing, `pip install -U yfinance` often helps, and the Stooq
  > fallback means the build still works regardless.

---

## Publishing to the web

This project is set up for **GitHub Pages**, serving the `docs/` folder.

One time setup: in the repository, open **Settings, then Pages**, choose **Deploy from
a branch**, and select the **main** branch with the **/docs** folder. Pages then gives
you the public URL.

To update the live site after a new statement:

```bash
python build.py --real
git add .
git commit -m "Refresh dashboard"
git push
```

GitHub Pages redeploys automatically on every push, so the rebuild and the push are the
only manual steps. There is no backend or database.

> Note: the analysis is **not** automatic yet. A new statement shows up on the site only
> after you run the steps above. Full automation would mean a scheduled GitHub Actions
> workflow that runs `python build.py --real` and pushes on its own. That is possible to
> add but is not set up in this repo.

---

Inspired by Apollo / Torsten Sløk. Original article:
https://www.apollo.com/wealth/the-daily-spark/the-number-of-words-in-the-fomc-statement-likely-going-down
