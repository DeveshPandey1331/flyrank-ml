# Capstone Report — Machine Learning-Based Content Refresh Opportunity Scoring

- **Author:** Devesh Pandey
- **Lane:** Refresh / Content Opportunity Scoring (Content Refresh)
- **Repo:** https://github.com/DeveshPandey1331/flyrank-ml
- **Date:** 2026-08-21

## 0. Abstract

Which of thousands of website pages should a content team review first for a refresh? Using FlyRank's anonymized 30,000-page starter release, I built a proxy label for decline (`trend_direction == "down"`), engineered a leakage-safe feature set from traffic, engagement, position, and content signals, and compared a transparent rule-based baseline against three models on a client-holdout split. Random Forest was the best performer, lifting Precision@50 from 0.24 (baseline rule) to 0.74 against a 0.542 base rate — roughly tripling how many of the top 50 flagged pages are genuinely declining. The output is a ranked, reason-coded refresh queue meant to support a human reviewer's decisions, not replace them.

## 1. Problem Framing

**Unit of analysis:** one row = one (anonymized content page, anonymized client) pair.

**Output:** a ranked score (0–100) per page, with a confidence label and a suggested action.

**Decision this supports:** given a large set of pages, which ones should be reviewed first for a content refresh? An SEO/content owner with limited review hours acts on the output — they work down the ranked queue as far as their time allows, rather than treating it as a single yes/no gate.

**Cost of a wrong call:** asymmetric. A false positive wastes review time on a page that didn't need it. A false negative leaves a genuinely declining page unreviewed, and its visibility keeps eroding while nobody looks at it. That asymmetry, plus the fact review time is the real constraint (not a fixed decision threshold), is why this is framed as a ranking problem rather than binary classification used in isolation.

**Why ML helps here:** page performance depends on many interacting signals (visibility, freshness, position, content depth, engagement) whose combined effect on decline risk isn't capturable by a short list of if/else rules — a hand-written rule using only 4 signals gets Precision@50 = 0.24, barely above the 0.542 base rate for "declining," while a model that can weigh and combine ~50 signals more than triples that number.

## 2. Data Safety

**Data used:** `data/raw/content_refresh_anonymized.csv` — 30,000 rows, trailing 90-day performance signals (impressions, clicks, sessions, engagement) plus content metadata (age, word count, content type, search intent). Ships anonymized under the internship's data-use terms — no client names, domains, URLs, titles, or keywords are present in the source file.

**Deliberately excluded / not used:** the gated full warehouse release (~79M rows, Hugging Face) — the 30k sample was sufficient to answer this question end-to-end and keeps the notebook runnable without a gated token.

**Leakage risks considered:**
- `trend_direction` and `trend_pct` are **label-derived** fields — the label (`is_declining_label`) is built directly from `trend_direction`. Both are excluded from every model feature. `trend_direction` is used only afterward, to generate human-readable reason codes (e.g. `declining_with_demand`) on top of the model's own probability — never as a training input.
- `client_id` is used **only for grouping** in the client-holdout split (Section 5) — it is never a model feature, since client identity itself would leak information about which rows share a site.
- `content_id` and `client_id` are opaque anonymized hashes, not features, and are only carried through the pipeline as row identifiers for the output queue.

**Confirmed:** nothing client-identifying (name, domain, URL, title, keyword) appears anywhere in `work/` — the source CSV itself contains none of these, so no additional stripping was required.

## 3. Baseline

A transparent, hand-written weighted score:

- 40% visibility (percentile rank of log impressions over 90 days)
- 30% freshness risk (percentile rank of days since last update)
- 25% position opportunity (room to move up in ranking, weighted by visibility)
- 5% content-depth gap (thin content, weighted by visibility)

`trend_direction` is deliberately **excluded** from the baseline score itself (only used for reason codes) so the baseline and the model are compared on the same footing — neither one gets to see the label-derived field as an input.

**Baseline result on the held-out test split, same metric as the model:**

| Metric | Value |
|---|---:|
| Base rate (declining, majority class) | 0.542 |
| ROC-AUC | 0.627 |
| Average Precision | 0.468 |
| Precision@50 | 0.24 |

The baseline's Precision@50 (0.24) sits **below** the 0.542 base rate — meaning the hand-written rule's top-50 picks are actually worse than a random sample of pages at finding declining ones. This is the honest number the model needs to beat, and it's a big part of why "the model is the capstone" here.

## 4. Model / Analysis

**Method:** Logistic Regression (interpretable linear baseline), Decision Tree (captures nonlinear splits), and Random Forest (captures feature interactions, generally the strongest of the three for tabular data with mixed types) — standard choices for a binary scoring/ranking task on structured, mixed numeric + categorical data. Random Forest was selected as primary based on Precision@50.

**Target / proxy, in one sentence:** `is_declining_label = 1` when `trend_direction == "down"`, a **current-window proxy for decline**, not a forecast of future performance.

**Feature list (leakage-safe):**
- *Numeric (18):* search volume, competition, CPC, word count, char count, log-scaled impressions/clicks/sessions/AI-sessions (90d), days with impressions, days with sessions, content age (days), days since last update, CTR, average position, engagement rate, scroll rate, AI-traffic share.
- *Categorical (8, one-hot encoded):* competition level, content type, main search intent, age tier, freshness tier, word-count tier, impression tier, position tier.

**Left out on purpose:** `trend_direction`, `trend_pct` (label-derived — see Section 2), `content_id`, `client_id` (identifiers, not signals).

## 5. Evaluation

**Split:** client-holdout, not a random row split. 20% of the 32 unique clients (by `client_id`) — 6 clients, 2,325 rows — are held out entirely for testing; the remaining 26 clients (27,675 rows) are used for training. This is the honest split for this question because it tests whether the model generalizes to a genuinely new site, not just to new pages from a site it already partly saw in training.

**Model vs. baseline, same split, same metrics:**

| Model | ROC-AUC | Avg. Precision | Precision@50 | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline (rules) | 0.627 | 0.468 | 0.24 | – | – |
| Logistic Regression | 0.700 | 0.522 | 0.40 | 0.567 | 0.566 |
| Decision Tree | 0.742 | 0.575 | 0.50 | 0.716 | 0.634 |
| **Random Forest (best)** | **0.750** | **0.617** | **0.74** | 0.748 | 0.641 |

Base rate for reference: **0.542** (declining pages make up 54.2% of the 30,000-row sample). Random Forest's ROC-AUC (0.750) and Precision@50 (0.74) both clear the base rate comfortably; the baseline rule's Precision@50 (0.24) does not.

**Error analysis:** Random Forest's recall (0.748) is noticeably higher than its precision (0.560 at the 0.5 threshold), meaning it leans toward flagging pages as declining — it catches most true declines but pulls in some healthy pages too. At the top of the ranked queue (Precision@50 = 0.74) this concentration of false positives drops off — errors cluster more toward the middle of the ranked list, not the top, which is exactly where a reviewer working down the queue would want the pain concentrated.

## 6. Interpretation

**In plain words, what the model found:** pages with intermittent or low visibility (`days_with_impressions`, `log_impressions_90d`), worse average search position (`avg_position`), and older content (`content_age_days`) were the strongest signals of decline risk — together they account for roughly half of the Random Forest's total feature importance. Content shape (word count, char count) and click-through behavior (CTR, log clicks) contributed a smaller but consistent share.

**Surprise / negative result:** competition level and CPC — signals that sound intuitively important for SEO — ranked near the bottom of feature importance. Decline risk in this sample is driven far more by *how consistently and visibly a page shows up* than by *how competitive its keyword space is*. This is a legitimate negative result worth stating plainly rather than omitting.

## 7. Recommendation

Ranked actions a FlyRank editor could use tomorrow, working down the queue in confidence order:

1. **`refresh_and_review_ctr` + high confidence** — real search demand, not converting to clicks; refresh content and review title/meta description together.
2. **`refresh` + high confidence** — stale, declining pages needing a straightforward content update.
3. **`expand_and_refresh`** — thin content with real traffic; expanding coverage is the likely lever, not just rewording.
4. **`monitor`** — low confidence or low visibility; not worth reviewer time this cycle, re-score next cycle.

**Confidence and limits, stated explicitly:** this is decision support, evaluated on a 30,000-row anonymized sample with a client-holdout split — it is *observed / directional*, not a guarantee. It says nothing about Google's ranking algorithm and makes no causal claim that refreshing a flagged page will improve its performance. A human should verify each high-confidence pick before acting on it.

## 8. Reproducibility

**Fresh-clone commands:**

```bash
git clone https://github.com/DeveshPandey1331/flyrank-ml
cd flyrank-ml
pip install -r requirements.txt
# Open and run work/notebooks/capstone.ipynb top to bottom (Colab or local Jupyter)
```

**Random seed:** `random_state=42` used throughout (client-holdout split and all three models) for reproducibility.

**Environment (this run):**
```
scikit-learn==1.8.0
pandas==3.0.2
numpy==2.4.4
matplotlib==3.10.8
```

**Evaluation is a single client-holdout split, not a sealed/blind holdout** — the split is built and evaluated once per notebook run inside `work/notebooks/capstone.ipynb` (Sections 3–4 of the notebook), and its output (`work/outputs/capstone_refresh_queue.csv`, plus the results table rendered in Section 4) is committed alongside the notebook, so the numbers above are checkable by re-running the notebook, not taken on faith.

## 9. Acknowledgments & Data Credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).
