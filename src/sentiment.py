"""
Hawkish / Dovish / Neutral stance of the LATEST FOMC statement.

Model-agnostic: works with any 3-class FOMC stance classifier. Currently set to a
DistilBERT classifier as a stopgap while access to gtfintechlab/FOMC-RoBERTa
(Shah, Paturi & Chava, "Trillion Dollar Words", ACL 2023) is pending — switch
MODEL_NAME back when it's granted. Labels are resolved safely: readable labels are
normalized, generic LABEL_n need an explicit entry in _LABEL_OVERRIDES, and an
unknown label raises rather than guessing (so hawk/dove can never be flipped).

The classifier is sentence-level, so we split the statement, classify each
sentence, and aggregate into a distribution + a net hawk-dove score. Heavy deps
(torch, transformers) are imported lazily. Nothing here is ever faked: if the
model can't run, the caller writes no sentiment file and the dashboard omits the
section.
"""

from __future__ import annotations

import re

# Active model. TEMPORARY: a DistilBERT FOMC classifier while access to
# gtfintechlab/FOMC-RoBERTa is pending. Switch back to "gtfintechlab/FOMC-RoBERTa"
# (one line) once the gated request is approved.
MODEL_NAME = "achen0525/DistilBERT_FOMC_Classifier"

# Bump when the net-score -> label band logic changes, so cached results get
# relabeled (without re-running the model). v2: neutral band widened to ±0.10.
LABEL_LOGIC_VERSION = 2

# Most fine-tuned models emit human-readable labels (Hawkish/Dovish/Neutral),
# which we normalize automatically. Models that emit generic LABEL_0/1/2 carry a
# model-specific meaning, so each such model MUST be listed here — we never guess
# a mapping, because flipping hawk <-> dove would silently mislead.
_LABEL_OVERRIDES = {
    # Verified (Trillion Dollar Words, ACL 2023):
    "gtfintechlab/FOMC-RoBERTa": {"LABEL_0": "Dovish", "LABEL_1": "Hawkish",
                                  "LABEL_2": "Neutral"},
    # Determined empirically (clearly-hawkish -> LABEL_1, dovish -> LABEL_0,
    # neutral -> LABEL_2). Same encoding as the Trillion Dollar Words dataset.
    "achen0525/DistilBERT_FOMC_Classifier": {"LABEL_0": "Dovish", "LABEL_1": "Hawkish",
                                             "LABEL_2": "Neutral"},
}


def _to_class(raw_label: str) -> str:
    """
    Map a model's raw label to Dovish / Hawkish / Neutral. Readable labels are
    normalized by keyword; generic LABEL_n are resolved via _LABEL_OVERRIDES.
    Raises (never guesses) if a label can't be resolved, so the dashboard can
    never show a mislabeled stance.
    """
    s = str(raw_label).strip()
    low = s.lower()
    if "hawk" in low:
        return "Hawkish"
    if "dov" in low:
        return "Dovish"
    if "neutral" in low:
        return "Neutral"
    mapped = _LABEL_OVERRIDES.get(MODEL_NAME, {}).get(s)
    if mapped:
        return mapped
    raise ValueError(
        f"Cannot map label '{raw_label}' from {MODEL_NAME} to hawkish/dovish/neutral. "
        f"Add its mapping to _LABEL_OVERRIDES in sentiment.py (refusing to guess)."
    )


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no NLTK download needed)."""
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text)
    return [s.strip() for s in parts if len(s.split()) >= 4]


def _load_classifier():
    """Lazy-build the HF text-classification pipeline. Raises if deps/model absent."""
    from transformers import (AutoConfig, AutoModelForSequenceClassification,
                              AutoTokenizer, pipeline)
    # Use each model's own config (labels, num_labels) — don't force a count, so
    # this works for any 3-class FOMC classifier, not just one architecture.
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("text-classification", model=model, tokenizer=tok, config=config,
                    framework="pt", device=-1)


def _headline_label(net: float) -> str:
    if abs(net) <= 0.10:
        return "Neutral"
    if net >= 0.25:
        return "Hawkish"
    if net > 0:
        return "Hawkish-leaning"
    if net <= -0.25:
        return "Dovish"
    return "Dovish-leaning"


def aggregate(sentences: list[str], preds: list[dict]) -> dict:
    """Turn per-sentence predictions into a statement-level summary."""
    rows = []
    for s, p in zip(sentences, preds):
        rows.append((s, _to_class(p["label"]), float(p.get("score", 0.0))))
    n = len(rows)
    counts = {"Dovish": 0, "Hawkish": 0, "Neutral": 0}
    for _, lab, _ in rows:
        counts[lab] = counts.get(lab, 0) + 1
    pct = {k: round(100 * v / n, 1) for k, v in counts.items()}
    net = round((counts["Hawkish"] - counts["Dovish"]) / n, 3)

    def top(label: str):
        cands = [(s, sc) for s, lab, sc in rows if lab == label]
        return max(cands, key=lambda x: x[1])[0] if cands else None

    return {
        "n_sentences": n,
        "counts": counts,
        "pct": pct,
        "net_score": net,
        "label": _headline_label(net),
        "label_logic_version": LABEL_LOGIC_VERSION,
        "example_hawkish": top("Hawkish"),
        "example_dovish": top("Dovish"),
    }


def classify_statement(text: str, classifier=None) -> dict:
    """Classify one statement's text. `classifier` injectable for testing."""
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("no classifiable sentences in statement text")
    if classifier is None:
        classifier = _load_classifier()
    preds = classifier(sentences, batch_size=16, truncation="only_first")
    return aggregate(sentences, preds)
