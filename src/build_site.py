"""
Site builder.

Renders the standalone, deployable index.html:
  1. Word count per meeting over time (bars, colored by chair).
  2. A scatter of statement length vs. the max VIX in the prior 6 weeks, with raw
     and detrended correlations.
  3. An Apollo-style dual-axis line chart of word count and Max VIX over time.
  4. An interactive word-cloud explorer: click a chair, see that era's signature
     FOMC language (wordcloud2.js, colored in the chair's accent).

Output is one self-contained file (assets via CDN) -> drop on GitHub Pages /
Netlify / any static host. No server required.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs, get_plotlyjs_version

from data_sources import CHAIR_COLORS, CHAIRS, DATA_DIR
from process import build_meeting_table

# GitHub Pages serves a static folder; "docs/" on the main branch is supported
# directly, so the build writes there.
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

ACCENT = "#B5161C"   # VIX / volatility
NAVY = "#2F4858"     # secondary cool tone
INK = "#1A1714"
MUTED = "#6B6256"
PAPER = "#F4EFE6"
RULE = "#D8CFC0"


def _base_layout(**over):
    """Shared editorial layout; callers override axes/legend/etc."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Newsreader, Georgia, serif", color=INK, size=13),
        margin=dict(l=64, r=28, t=58, b=54),
        hoverlabel=dict(bgcolor=PAPER, bordercolor=RULE,
                        font=dict(family="IBM Plex Mono, monospace", size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    base.update(over)
    return base


def _era_bands(df: pd.DataFrame):
    """Shaded chair-era background bands + chair labels for the time axis."""
    xmin, xmax = df["date"].min(), df["date"].max()
    shapes, annotations = [], []
    for i, c in enumerate(CHAIRS):
        start = pd.Timestamp(c.start)
        end = pd.Timestamp(c.end) if c.end else xmax
        if end < xmin or start > xmax:
            continue
        start = max(start, xmin)
        end = min(end, xmax)
        shapes.append(dict(
            type="rect", xref="x", yref="paper", x0=start, x1=end, y0=0, y1=1,
            fillcolor=CHAIR_COLORS[c.name], opacity=0.05 if i % 2 == 0 else 0.09,
            line_width=0, layer="below",
        ))
        annotations.append(dict(
            x=start + (end - start) / 2, y=1.04, xref="x", yref="paper",
            text=c.name.upper(), showarrow=False,
            font=dict(family="IBM Plex Mono, monospace", size=10, color=CHAIR_COLORS[c.name]),
        ))
    return shapes, annotations


def build_timeline_figure(df: pd.DataFrame) -> go.Figure:
    """
    Hero view: how long each statement ran, over time. One metric, one axis —
    bars colored by the sitting chair, so the era story reads at a glance with no
    scale conflict.
    """
    fig = go.Figure()
    fig.add_bar(
        x=df["date"], y=df["word_count"], name="Statement length",
        marker=dict(color=[CHAIR_COLORS.get(c, "#888") for c in df["chair"]], line=dict(width=0)),
        opacity=0.92,
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:,} words<extra></extra>",
    )
    shapes, annotations = _era_bands(df)

    fig.update_layout(**_base_layout(
        shapes=shapes, annotations=annotations, barmode="overlay", bargap=0.2,
        hovermode="x unified", showlegend=False,
        xaxis=dict(showgrid=False, color=MUTED, linecolor=RULE, ticks="outside", tickcolor=RULE),
        yaxis=dict(title=dict(text="WORDS PER STATEMENT", font=dict(size=11, color=MUTED)),
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False, color=MUTED),
    ))
    return fig


def build_scatter_figure(df: pd.DataFrame, *, x_col: str, x_title: str,
                         hover_x: str, x_log: bool = False) -> go.Figure:
    """
    Relationship view: each meeting is a dot — statement length vs. the chosen
    market measure — colored by chair, with an OLS trend line and Pearson r. This
    is the honest way to show "does X relate to statement length?".
    """
    act = df.copy()
    # Keep only finite (x, y) so np.polyfit / corrcoef don't choke on NaNs.
    act = act[np.isfinite(act[x_col].to_numpy(dtype=float))
              & np.isfinite(act["word_count"].to_numpy(dtype=float))]
    if x_log:
        act = act[act[x_col] > 0]

    fig = go.Figure()
    for c in CHAIRS:
        sub = act[act["chair"] == c.name]
        if not len(sub):
            continue
        fig.add_scatter(
            x=sub[x_col], y=sub["word_count"], mode="markers", name=c.name,
            text=sub["date"].dt.strftime("%b %Y"),
            marker=dict(color=CHAIR_COLORS[c.name], size=7, opacity=0.55,
                        line=dict(color=PAPER, width=0.5)),
            hovertemplate="<b>" + c.name + "</b> · %{text}<br>" + hover_x +
                          "<br>%{y:,} words<extra></extra>",
        )

    # OLS trend (fit in log-x space when the axis is logarithmic). Only fit when
    # there are enough points with actual spread in x.
    x = act[x_col].to_numpy(dtype=float)
    y = act["word_count"].to_numpy(dtype=float)
    xs = np.log(x) if x_log else x
    r_text = None
    if len(xs) >= 3 and np.ptp(xs) > 0:
        slope, intercept = np.polyfit(xs, y, 1)
        grid = np.linspace(xs.min(), xs.max(), 60)
        fig.add_scatter(
            x=np.exp(grid) if x_log else grid, y=slope * grid + intercept,
            mode="lines", name="Trend", line=dict(color=INK, width=1.5, dash="dash"),
            hoverinfo="skip",
        )
        r = float(np.corrcoef(xs, y)[0, 1])

        # Detrended correlation: correlate the change between consecutive meetings
        # (first differences). This removes the slow era/regime drift in statement
        # length and isolates the short-term "did volatility and wordiness move
        # together?" relationship Apollo is really about.
        ordered = act.sort_values("date")
        ox = ordered[x_col].to_numpy(dtype=float)
        oxs = np.log(ox) if x_log else ox
        dvix = np.diff(oxs)
        dwords = np.diff(ordered["word_count"].to_numpy(dtype=float))
        r_detr = (float(np.corrcoef(dvix, dwords)[0, 1])
                  if len(dwords) >= 3 and np.ptp(dvix) > 0 else float("nan"))

        if np.isfinite(r):
            r_text = f"raw  r = {r:+.2f}   ·   n = {len(xs)}"
            if np.isfinite(r_detr):
                r_text += f"<br>detrended  Δr = {r_detr:+.2f}"

    annotations = []
    if r_text:
        annotations.append(dict(
            x=0.025, y=0.97, xref="paper", yref="paper", xanchor="left", yanchor="top",
            text=r_text, showarrow=False, align="left",
            font=dict(family="IBM Plex Mono, monospace", size=13, color=INK),
            bgcolor=PAPER, bordercolor=RULE, borderwidth=1, borderpad=6,
        ))

    fig.update_layout(**_base_layout(
        hovermode="closest",
        annotations=annotations,
        xaxis=dict(title=dict(text=x_title, font=dict(size=11, color=MUTED)),
                   type="log" if x_log else "linear",
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False,
                   color=MUTED, linecolor=RULE, ticks="outside", tickcolor=RULE),
        yaxis=dict(title=dict(text="WORDS PER STATEMENT", font=dict(size=11, color=MUTED)),
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False, color=MUTED),
    ))
    return fig


def build_vix_figure(df: pd.DataFrame) -> go.Figure:
    return build_scatter_figure(
        df, x_col="vix_prior_6w",
        x_title="Max VIX in the 6 weeks before each meeting",
        hover_x="Max VIX: %{x:.1f}",
    )


def build_apollo_figure(df: pd.DataFrame) -> go.Figure:
    """
    Apollo-style view: word count and Max-VIX-prior-6w as two lines over time on
    a dual axis (raw values, not normalized). Shows WHERE the two co-move — the
    crisis spikes — which the scatter can't. Our palette: ink for word count,
    accent red for the VIX.
    """
    d = df.sort_values("date")
    fig = go.Figure()
    fig.add_scatter(
        x=d["date"], y=d["word_count"], name="Statement word count", mode="lines",
        line=dict(color=INK, width=2), yaxis="y",
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} words<extra></extra>",
    )
    fig.add_scatter(
        x=d["date"], y=d["vix_prior_6w"], name="Max VIX, prior 6 weeks", mode="lines",
        line=dict(color=ACCENT, width=1.6), yaxis="y2",
        hovertemplate="Max VIX: %{y:.0f}<extra></extra>",
    )

    shapes, annotations = _era_bands(df)
    # Apollo-style vertical dividers at each chair handover.
    xmin, xmax = df["date"].min(), df["date"].max()
    for c in CHAIRS:
        start = pd.Timestamp(c.start)
        if xmin < start < xmax:
            shapes.append(dict(type="line", xref="x", yref="paper",
                               x0=start, x1=start, y0=0, y1=1,
                               line=dict(color=INK, width=1, dash="dot")))

    fig.update_layout(**_base_layout(
        shapes=shapes, annotations=annotations, hovermode="x unified",
        xaxis=dict(showgrid=False, color=MUTED, linecolor=RULE, ticks="outside", tickcolor=RULE),
        yaxis=dict(title=dict(text="WORDS PER STATEMENT", font=dict(size=11, color=INK)),
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False,
                   color=INK, rangemode="tozero"),
        yaxis2=dict(title=dict(text="MAX VIX", font=dict(size=11, color=ACCENT)),
                    overlaying="y", side="right", showgrid=False, zeroline=False,
                    color=ACCENT, tickfont=dict(color=ACCENT), rangemode="tozero"),
    ))
    return fig


def build_rates_figure(rates: pd.DataFrame, df: pd.DataFrame) -> go.Figure:
    """
    Fed funds rate vs. the 2-year Treasury yield over time — both in percent on one
    axis. The gap is a market-implied read on the policy path (2Y above funds rate
    = hikes priced; below = cuts), which pairs with the statement-tone section.
    """
    r = rates.sort_values("date")
    fig = go.Figure()
    fig.add_scatter(
        x=r["date"], y=r["fed_funds"], name="Fed funds rate", mode="lines",
        line=dict(color=INK, width=2),
        hovertemplate="<b>%{x|%b %Y}</b><br>Fed funds: %{y:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=r["date"], y=r["dgs2"], name="2-year Treasury", mode="lines",
        line=dict(color=ACCENT, width=1.8),
        hovertemplate="2-year: %{y:.2f}%<extra></extra>",
    )
    shapes, annotations = _era_bands(df)
    xmin, xmax = df["date"].min(), df["date"].max()
    for c in CHAIRS:
        start = pd.Timestamp(c.start)
        if xmin < start < xmax:
            shapes.append(dict(type="line", xref="x", yref="paper", x0=start, x1=start,
                               y0=0, y1=1, line=dict(color=INK, width=1, dash="dot")))
    fig.update_layout(**_base_layout(
        shapes=shapes, annotations=annotations, hovermode="x unified",
        xaxis=dict(showgrid=False, color=MUTED, linecolor=RULE, ticks="outside", tickcolor=RULE),
        yaxis=dict(title=dict(text="PERCENT", font=dict(size=11, color=MUTED)),
                   showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False,
                   color=MUTED, ticksuffix="%", rangemode="tozero"),
    ))
    return fig


def build_sentiment_figure(s: dict) -> go.Figure:
    """Bar chart: share of sentences that are dovish / neutral / hawkish."""
    cats = [("Dovish", NAVY), ("Neutral", MUTED), ("Hawkish", ACCENT)]
    names = [c for c, _ in cats]
    pct = [float(s["pct"].get(c, 0)) for c in names]
    counts = [int(s["counts"].get(c, 0)) for c in names]
    colors = [col for _, col in cats]

    fig = go.Figure()
    fig.add_bar(
        x=names, y=pct,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.0f}%" for p in pct], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK, family="IBM Plex Mono, monospace", size=14),
        customdata=counts,
        hovertemplate="%{x}: %{customdata} sentences (%{y:.0f}%)<extra></extra>",
        width=0.6,
    )
    fig.update_layout(**_base_layout(
        showlegend=False, hovermode="closest",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, color=INK, linecolor=RULE, ticks="outside",
                   tickcolor=RULE, tickfont=dict(family="IBM Plex Mono, monospace", size=13)),
        yaxis=dict(title=dict(text="% OF SENTENCES", font=dict(size=11, color=MUTED)),
                   range=[0, max(pct + [10]) * 1.18], showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", zeroline=False, color=MUTED),
    ))
    return fig


def _to_div(fig, div_id: str, first: bool):
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False,
        div_id=div_id, config={"displayModeBar": False, "responsive": True},
    )


def _section(title: str, note: str, chart_html: str) -> str:
    return (f'  <section>\n'
            f'    <div class="sec-head"><h2>{title}</h2></div>\n'
            f'    <p class="sec-note">{note}</p>\n'
            f'    <div class="card">{chart_html}</div>\n'
            f'  </section>')


def render(df: pd.DataFrame) -> Path:
    if df.empty:
        raise ValueError("render(): empty meeting table — no market data to plot. "
                         "Fix the market fetch or run with --sample.")
    DOCS_DIR.mkdir(exist_ok=True)

    # One pinned Plotly script in <head>, with a Subresource Integrity hash computed
    # from the exact plotly.js bytes this build bundles — so a compromised CDN can't
    # swap in malicious code. cdn.plot.ly serves the same bytes for this version, so
    # the hash matches regardless of the local Plotly version. Figures are emitted
    # with include_plotlyjs=False and rely on this global.
    _pjs = get_plotlyjs().encode("utf-8")
    _psri = "sha384-" + base64.b64encode(hashlib.sha384(_pjs).digest()).decode()
    plotly_script = (
        f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" '
        f'integrity="{_psri}" crossorigin="anonymous" referrerpolicy="no-referrer">'
        f'</script>'
    )

    # Countdown to the next FOMC statement. The schedule (statement release at
    # 2 p.m. ET on day two, with the correct EDT/EST offset per meeting) is the
    # Fed's published calendar through Jan 2028; the script picks the first date
    # still ahead of the viewer's clock and rolls forward on its own as meetings
    # pass, falling back to a neutral message once the list runs out.
    fomc_dates = [
        "2026-06-17T14:00:00-04:00", "2026-07-29T14:00:00-04:00",
        "2026-09-16T14:00:00-04:00", "2026-10-28T14:00:00-04:00",
        "2026-12-09T14:00:00-05:00", "2027-01-27T14:00:00-05:00",
        "2027-03-17T14:00:00-04:00", "2027-04-28T14:00:00-04:00",
        "2027-06-09T14:00:00-04:00", "2027-07-28T14:00:00-04:00",
        "2027-09-15T14:00:00-04:00", "2027-10-27T14:00:00-04:00",
        "2027-12-08T14:00:00-05:00", "2028-01-26T14:00:00-05:00",
    ]
    countdown_script = (
        "<script>(function(){var M=" + json.dumps(fomc_dates) + ";"
        "var d=document.getElementById('cd-date'),t=document.getElementById('cd-timer');"
        "if(!d||!t)return;"
        "function p(n){return String(n).padStart(2,'0');}"
        "function nx(){var now=Date.now();for(var i=0;i<M.length;i++){var x=new Date(M[i]).getTime();if(x>now)return x;}return null;}"
        "function fmt(x){return new Date(x).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'America/New_York'});}"
        "function tick(){var g=nx();if(g===null){d.textContent='Schedule to be announced';t.textContent='';return;}"
        "d.textContent=fmt(g);var s=Math.max(0,g-Date.now());"
        "var dd=Math.floor(s/864e5),hh=Math.floor(s%864e5/36e5),mm=Math.floor(s%36e5/6e4),ss=Math.floor(s%6e4/1e3);"
        "t.textContent=dd+'d '+p(hh)+':'+p(mm)+':'+p(ss);}"
        "tick();setInterval(tick,1000);})();</script>"
    )

    timeline_note = (
        "Every statement since 1994, one bar per meeting, sized by word count and "
        "colored by the sitting chair. Short under Greenspan, ballooning through the "
        "crisis and Yellen years, easing under Powell, and now Warsh. A new chair "
        "appears once a real statement exists for them."
    )

    # (title, note, figure, div_id) — order defines the section numbering.
    specs = [
        ("How long are Fed statements?", timeline_note,
         build_timeline_figure(df), "timeline-chart"),
        ("Statements vs. the VIX",
         "Each dot is one meeting: the peak VIX (a gauge of expected US stock market "
         "volatility) in the six weeks before it (horizontal) against the statement's "
         "length (vertical), following Apollo's original chart. The box reports two "
         "correlations: the raw figure across the whole sample, held down by the long "
         "statements of the calm Yellen years, and a detrended figure that compares the "
         "change from one meeting to the next, stripping out that era drift to isolate "
         "the immediate link between volatility and length.",
         build_vix_figure(df), "vix-chart"),
        ("Statements and the VIX, Apollo view",
         "Statement word count (ink, left axis) and the peak VIX in the six weeks before "
         "each meeting (red, right axis), over time. It shows <em>where</em> the two move "
         "together, the crisis spikes of 2008 and 2020, even though, as the scatter "
         "makes clear, the overall relationship is weak. Dotted lines mark each change "
         "of chair.",
         build_apollo_figure(df), "apollo-chart"),
    ]

    # Fed Funds vs 2Y (FRED). Optional — present only if rates.csv exists. Sits
    # right before the tone section so the two can be read together.
    rates_path = DATA_DIR / "rates.csv"
    if rates_path.exists():
        try:
            rates = pd.read_csv(rates_path, parse_dates=["date"])
        except Exception:
            rates = None
        if rates is not None and not rates.empty:
            specs.append((
                "The federal funds rate and the two year Treasury",
                "The federal funds rate (ink) against the two year Treasury yield (red), "
                "both in percent. The two year yield is largely the market's bet on where "
                "the fed funds rate is headed over the coming two years, so when it sits "
                "<em>above</em> the funds rate, markets are pricing rate rises (a hawkish "
                "path); when <em>below</em>, cuts. Read it alongside the latest "
                "statement's tone: does the Fed's language match where the market thinks "
                "rates are going?",
                build_rates_figure(rates, df), "rates-chart"))

    # Latest-statement sentiment (real only; never fabricated; absent in sample).
    sent = None
    sent_path = DATA_DIR / "sentiment.json"
    if sent_path.exists():
        try:
            _s = json.loads(sent_path.read_text())
            if _s.get("n_sentences"):
                sent = _s
        except Exception:
            sent = None
    sentiment_section = ""  # rendered last, after the word cloud (see template)
    if sent:
        s = sent
        # The label is a pure function of net_score, so derive it fresh here — that
        # way tuning the neutral band takes effect on the next build without
        # re-running the model or clearing the cached sentiment.json.
        try:
            import sentiment as _snt
            s["label"] = _snt._headline_label(float(s["net_score"]))
        except Exception:
            pass
        model = s.get("model", "")
        if "FOMC-RoBERTa" in model:
            method = ('FOMC-RoBERTa, the RoBERTa-large classifier from the '
                      '<a href="https://aclanthology.org/2023.acl-long.368/" '
                      'target="_blank" rel="noopener">Trillion Dollar Words</a> paper '
                      '(Shah, Paturi, and Chava, ACL 2023)')
        else:
            method = "a sentence level FOMC stance classifier"
        note = (
            f"The latest statement, dated {s['statement_date']}, is classified sentence "
            f"by sentence with {method}, then aggregated. Its net score of "
            f"{s['net_score']:+.2f} (a scale where negative is dovish, positive "
            f"hawkish) reads as <strong>{s['label']}</strong> across "
            f"{s['n_sentences']} sentences."
        )
        ex_rows = ""
        if s.get("example_hawkish"):
            ex_rows += ('<p class="tone-ex"><span class="tone-tag hawk">Most hawkish</span>'
                        f'<span class="tone-q">“{s["example_hawkish"]}”</span></p>')
        if s.get("example_dovish"):
            ex_rows += ('<p class="tone-ex"><span class="tone-tag dove">Most dovish</span>'
                        f'<span class="tone-q">“{s["example_dovish"]}”</span></p>')
        examples = f'\n    <div class="tone-examples">{ex_rows}</div>' if ex_rows else ""
        chart = _to_div(build_sentiment_figure(s), "sentiment-chart", first=False)
        sentiment_section = (
            '  <section>\n'
            '    <div class="sec-head"><h2>The tone of the latest statement</h2></div>\n'
            f'    <p class="sec-note">{note}</p>\n'
            f'    <div class="card">{chart}</div>{examples}\n'
            '  </section>'
        )

    sections_html = "\n".join(
        _section(title, note, _to_div(fig, div_id, first=(i == 0)))
        for i, (title, note, fig, div_id) in enumerate(specs)
    )

    freqs = json.loads((DATA_DIR / "freqs.json").read_text())
    meta = json.loads((DATA_DIR / "meta.json").read_text())

    # headline stats
    act = df
    latest = act.iloc[-1]
    peak = act.loc[act["word_count"].idxmax()]

    # Latest-statement tone replaces the old correlation stat. Real only — in
    # sample mode there's no sentiment, so the cell shows a neutral placeholder
    # (never faked).
    if sent:
        lab = sent["label"]
        tone_value = lab
        tone_sub = f"Latest tone · net {sent['net_score']:+.2f}"
        low = lab.lower()
        tone_color = ACCENT if "hawk" in low else NAVY if "dov" in low else MUTED
    else:
        tone_value = "n/a"
        tone_sub = "Latest tone · run --real"
        tone_color = MUTED

    chair_order = [c.name for c in CHAIRS if c.name in freqs]
    chair_colors_js = json.dumps({c: CHAIR_COLORS[c] for c in chair_order})

    tabs = "\n".join(
        f'<button class="chair-tab" data-chair="{c}" '
        f'style="--c:{CHAIR_COLORS[c]}">{c}</button>'
        for c in chair_order
    )

    src_note = ("Illustrative sample data. Run the pipeline with <code>--real</code> "
                "to populate live figures." if meta["source"] == "sample"
                else "Live data from the Federal Reserve and market sources.")

    html = TEMPLATE.format(
        chart_sections=sections_html,
        plotly_script=plotly_script,
        countdown_script=countdown_script,
        sentiment_section=sentiment_section,
        tabs=tabs,
        freqs_json=json.dumps(freqs),
        chair_colors_js=chair_colors_js,
        first_chair=chair_order[0],
        tone_value=tone_value,
        tone_sub=tone_sub,
        tone_color=tone_color,
        latest_words=f"{int(latest['word_count']):,}",
        latest_date=pd.Timestamp(latest["date"]).strftime("%b %Y"),
        peak_words=f"{int(peak['word_count']):,}",
        peak_chair=peak["chair"],
        n_meetings=f"{len(df):,}",
        src_note=src_note,
        generated=datetime.now().strftime("%d %b %Y"),
    )
    out = DOCS_DIR / "index.html"
    out.write_text(html)
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>The Length of Fed Words · FOMC Communication Monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/wordcloud@1.2.2/src/wordcloud2.js" integrity="sha384-qFO0W0pEXGHcEQcFOCL1dqVNS1/5JIj7kQpsg0oVND1YuikX5rrXdX0SubOXnmhJ" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
{plotly_script}
<style>
  :root{{
    --paper:#F4EFE6; --ink:#1A1714; --muted:#6B6256; --rule:#D8CFC0;
    --accent:#B5161C; --navy:#2F4858;
  }}
  *{{box-sizing:border-box}}
  body{{
    margin:0; background:var(--paper); color:var(--ink);
    font-family:"Newsreader",Georgia,serif; line-height:1.5;
    background-image:radial-gradient(circle at 1px 1px, rgba(0,0,0,0.025) 1px, transparent 0);
    background-size:22px 22px;
  }}
  .wrap{{max-width:1080px; margin:0 auto; padding:48px 28px 80px}}
  .masthead{{border-bottom:2px solid var(--ink); padding-bottom:14px; margin-bottom:8px;
    display:flex; justify-content:space-between; align-items:flex-end; gap:24px 40px; flex-wrap:wrap}}
  .mast-left{{min-width:0}}
  .countdown{{font-family:"IBM Plex Mono",monospace; text-align:right; padding-bottom:6px}}
  .cd-label{{font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:var(--muted)}}
  .cd-date{{font-family:"Fraunces",serif; font-size:18px; color:var(--ink); margin-top:3px}}
  .cd-timer{{font-size:23px; color:var(--accent); margin-top:3px;
    font-variant-numeric:tabular-nums; letter-spacing:.02em}}
  @media(max-width:640px){{ .countdown{{text-align:left}} }}
  .kicker{{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.22em;
    text-transform:uppercase; color:var(--accent); margin:0 0 10px}}
  h1{{font-family:"Fraunces",serif; font-weight:900; font-size:clamp(34px,6vw,62px);
    line-height:0.98; letter-spacing:-0.01em; margin:0}}
  .standfirst{{font-size:clamp(15px,2vw,19px); color:#3a342d; max-width:62ch;
    margin:18px 0 0; font-style:italic}}
  .meta-row{{display:flex; flex-wrap:wrap; gap:8px 26px; margin-top:18px;
    font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.06em;
    color:var(--muted); text-transform:uppercase}}
  .stats{{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:1px; background:var(--rule); border:1px solid var(--rule);
    margin:34px 0 26px}}
  .stat{{background:var(--paper); padding:16px 18px}}
  .stat .n{{font-family:"Fraunces",serif; font-weight:600; font-size:30px; line-height:1}}
  .stat .l{{font-family:"IBM Plex Mono",monospace; font-size:10px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted); margin-top:8px}}
  .stat.accent .n{{color:var(--accent)}}
  section{{margin-top:46px}}
  .sec-head{{display:flex; align-items:baseline; gap:14px; border-bottom:1px solid var(--rule);
    padding-bottom:8px; margin-bottom:8px}}
  .sec-head h2{{font-family:"Fraunces",serif; font-weight:600; font-size:26px; margin:0}}
  .sec-head .no{{font-family:"IBM Plex Mono",monospace; font-size:12px; color:var(--accent)}}
  .sec-note{{color:#4A443C; font-size:15px; max-width:64ch; margin:0 0 18px}}
  .sec-note a{{color:var(--accent); text-decoration:underline; text-underline-offset:2px}}
  .tone-examples{{margin-top:14px; display:grid; gap:9px}}
  .tone-ex{{display:flex; gap:12px; align-items:baseline; margin:0;
    font-family:"Newsreader",Georgia,serif}}
  .tone-tag{{flex:none; font-family:"IBM Plex Mono",monospace; font-size:10px;
    letter-spacing:.1em; text-transform:uppercase; color:#fff; padding:3px 8px; border-radius:2px}}
  .tone-tag.hawk{{background:var(--accent)}}
  .tone-tag.dove{{background:var(--navy)}}
  .tone-q{{font-style:italic; color:#3a342d; font-size:15px; line-height:1.45}}
  .card{{background:#FBF8F1; border:1px solid var(--rule); padding:14px}}
  #timeline-chart{{width:100%; height:440px}}
  #vix-chart, #apollo-chart, #rates-chart{{width:100%; height:440px}}
  #sentiment-chart{{width:100%; height:260px}}
  .cloud-layout{{display:grid; grid-template-columns:200px 1fr; gap:0;
    border:1px solid var(--rule); background:#FBF8F1}}
  .chair-rail{{border-right:1px solid var(--rule); padding:10px}}
  .chair-tab{{display:block; width:100%; text-align:left; background:none; border:none;
    cursor:pointer; font-family:"Fraunces",serif; font-size:20px; color:var(--muted);
    padding:11px 12px; border-left:3px solid transparent; transition:.15s}}
  .chair-tab:hover{{color:var(--ink); background:rgba(0,0,0,0.03)}}
  .chair-tab.active{{color:var(--c); border-left-color:var(--c);
    background:rgba(0,0,0,0.035); font-weight:600}}
  .cloud-stage{{position:relative; min-height:460px; padding:6px}}
  .cloud-ctrl{{display:flex; align-items:center; gap:12px; padding:8px 10px 6px;
    font-family:"IBM Plex Mono",monospace; font-size:10px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted)}}
  .cloud-ctrl input[type=range]{{flex:1; height:2px; cursor:pointer; accent-color:var(--accent)}}
  .cloud-ctrl .val{{min-width:5.5em; text-align:right; color:var(--ink); letter-spacing:.04em}}
  #cloud{{width:100%; height:410px; display:block}}
  .cloud-cap{{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.08em;
    text-transform:uppercase; color:var(--muted); padding:6px 10px}}
  footer{{margin-top:60px; border-top:1px solid var(--rule); padding-top:16px;
    font-family:"IBM Plex Mono",monospace; font-size:11px; color:var(--muted);
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px}}
  code{{background:rgba(0,0,0,0.05); padding:1px 5px; border-radius:3px; font-size:.92em}}
  a{{color:var(--accent)}}
  footer a{{color:var(--muted); text-decoration:underline; text-underline-offset:2px}}
  footer a:hover{{color:var(--accent)}}
  @media(max-width:680px){{.cloud-layout{{grid-template-columns:1fr}}
    .chair-rail{{border-right:none; border-bottom:1px solid var(--rule);
      display:flex; flex-wrap:wrap; gap:4px}}
    .chair-tab{{width:auto; font-size:16px; border-left:none; border-bottom:3px solid transparent}}
    .chair-tab.active{{border-left:none; border-bottom-color:var(--c)}}}}
</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <div class="mast-left">
      <p class="kicker">FOMC Communication Monitor</p>
      <h1>The Length of<br>Fed Words</h1>
    </div>
    <div class="countdown" aria-live="polite">
      <div class="cd-label">Next statement</div>
      <div class="cd-date" id="cd-date"></div>
      <div class="cd-timer" id="cd-timer"></div>
    </div>
  </header>
  <p class="standfirst">How long are the Federal Reserve's statements, and does length
  track the market's mood? Following every FOMC statement since 1994, this dashboard
  finds that length owes more to each chair's style than to volatility, the two aligning
  mainly in a crisis; it then sets the latest statement's hawkish or dovish tone against
  the rate path the market is pricing.</p>

  <div class="meta-row">
    <span>Statement length · peak VIX over six weeks · latest statement tone</span>
    <span>{n_meetings} meetings</span>
    <span>Updated {generated}</span>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="n" style="color:{tone_color}">{tone_value}</div>
      <div class="l">{tone_sub}</div>
    </div>
    <div class="stat">
      <div class="n">{peak_words}</div>
      <div class="l">Peak length · {peak_chair} era</div>
    </div>
    <div class="stat">
      <div class="n">{latest_words}</div>
      <div class="l">Latest · {latest_date}</div>
    </div>
  </div>

{chart_sections}

  <section>
    <div class="sec-head"><h2>The vocabulary of each chair</h2></div>
    <p class="sec-note">Each era leaned on its own language. Select a chair to see
    the words that defined their statements.</p>
    <div class="cloud-layout">
      <div class="chair-rail">{tabs}</div>
      <div class="cloud-stage">
        <div class="cloud-ctrl">
          <label for="word-slider">Words shown</label>
          <input type="range" id="word-slider" min="5" max="40" value="40" step="1">
          <span class="val" id="word-val"></span>
        </div>
        <canvas id="cloud"></canvas>
        <div class="cloud-cap" id="cloud-cap"></div>
      </div>
    </div>
  </section>

{sentiment_section}

  <footer>
    <span>{src_note}</span>
    <span><a href="https://www.linkedin.com/in/mbbedir/" target="_blank" rel="noopener">LinkedIn</a></span>
    <span><a href="https://www.apollo.com/wealth/the-daily-spark/the-number-of-words-in-the-fomc-statement-likely-going-down" target="_blank" rel="noopener">Inspired by Apollo / Torsten Sløk</a> · Built with Plotly</span>
  </footer>
</div>

<script>
  const FREQS = {freqs_json};
  const CHAIR_COLORS = {chair_colors_js};
  let activeChair = "{first_chair}";

  function shade(hex, amt) {{
    const n = parseInt(hex.slice(1),16);
    let r=(n>>16)+amt, g=((n>>8)&255)+amt, b=(n&255)+amt;
    r=Math.max(0,Math.min(255,r)); g=Math.max(0,Math.min(255,g)); b=Math.max(0,Math.min(255,b));
    return "rgb("+r+","+g+","+b+")";
  }}

  const slider = document.getElementById("word-slider");
  const wordVal = document.getElementById("word-val");

  function drawCloud(chair) {{
    const canvas = document.getElementById("cloud");
    const stage = canvas.parentElement;
    const cssW = stage.clientWidth - 12, cssH = 410;
    const dpr = window.devicePixelRatio || 1;
    // Work entirely in device pixels (no ctx.scale): this keeps wordcloud2's text
    // measurement and its packing grid in the SAME units, which is what stops words
    // from being clipped at the edges.
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";

    const all = Object.entries(FREQS[chair]).sort((a, b) => b[1] - a[1]);
    const n = Math.min(parseInt(slider.value, 10), all.length);
    const entries = all.slice(0, n);
    const maxW = Math.max(...entries.map(e => e[1]));
    const base = CHAIR_COLORS[chair];

    // Size the LARGEST word to the canvas (bounded), not an open-ended multiple of
    // its weight, so even the biggest term fits. Fewer words give a touch larger.
    const fill = Math.min(1.7, Math.sqrt(all.length / n));
    const maxFont = Math.min(cssH * 0.24, cssW * 0.13) * fill * dpr;

    WordCloud(canvas, {{
      list: entries,
      gridSize: Math.max(4, Math.round(8 * cssW / 600 * dpr)),
      weightFactor: function (v) {{ return Math.max(12 * dpr, v / maxW * maxFont); }},
      fontFamily: "Fraunces, Georgia, serif",
      color: function () {{ return shade(base, Math.floor(Math.random()*60) - 20); }},
      backgroundColor: "transparent",
      rotateRatio: 0.18,
      rotationSteps: 2,
      shrinkToFit: true,
      drawOutOfBound: false,
      origin: [canvas.width / 2, canvas.height / 2],
    }});
    wordVal.textContent = n + " of " + all.length;
    document.getElementById("cloud-cap").textContent =
      chair + " · showing top " + n + " terms";
  }}

  function selectChair(chair) {{
    activeChair = chair;
    const total = Object.keys(FREQS[chair]).length;
    slider.max = total;
    if (parseInt(slider.value, 10) > total) slider.value = total;
    slider.style.accentColor = CHAIR_COLORS[chair];
    document.querySelectorAll(".chair-tab").forEach(b =>
      b.classList.toggle("active", b.dataset.chair === chair));
    drawCloud(chair);
  }}

  slider.addEventListener("input", () => drawCloud(activeChair));

  document.querySelectorAll(".chair-tab").forEach(btn =>
    btn.addEventListener("click", () => selectChair(btn.dataset.chair)));

  let rT;
  window.addEventListener("resize", () => {{
    clearTimeout(rT); rT = setTimeout(() => drawCloud(activeChair), 200);
  }});

  window.addEventListener("load", () => selectChair(activeChair));
</script>
{countdown_script}
</body>
</html>
"""


if __name__ == "__main__":
    df = build_meeting_table()
    print("rendered:", render(df))
