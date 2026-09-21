# Frozen predictions

Written **2026-09-21, before a single API call was made** against scifact. Scored in
`README.md` after. Nothing here is edited once a run has started; a wrong prediction stays
visible and is the most useful row in the table.

Setup: BEIR scifact, 300 test queries, BM25 top-240 candidates per query (`bm25s`), reranked
by Jev `choice` (`jev-1.13.0`, pinned). Metric nDCG@10 against the published qrels. Chunk
size 60, so the reduction faces 4:1 compression — the same compression a 1,020-candidate
problem faces at the 255 cap.

| # | prediction | why | outcome |
|---|---|---|---|
| **P1** | At K=240 a flat call cannot rank: **≥90% of candidates come back at p=0.00**, so nDCG@10 is decided by ties. Quantisation, not the 255-option cap, is the real ceiling on wide reranking. | Jev returns probabilities at 2 decimal places (measured, 2,784 calls, systemone `docs/PROBES.md`). 240 options sharing unit mass at a 0.01 grid leaves almost every candidate at the floor. | |
| **P2** | Merging raw within-chunk probabilities (`naive`) loses **≥0.05 nDCG@10** against two-stage `final`. | A probability is normalised inside its own call. A chunk of 60 weak candidates hands its best one a large share; a chunk holding three strong ones splits mass three ways. The scales are not comparable and sorting across them compares nothing. | |
| **P3** | Additive anchor equating with 8 anchors lands **within 0.02 nDCG@10** of two-stage `final`, at one fewer round-trip. | If Jev is a conditional logit, each call's log-probabilities are the true utilities minus that call's log partition function. Shared items recover the offset — common-item equating, Kolen & Brennan. | |
| **P4** | Per-anchor implied offsets inside one chunk agree: **SD < 0.5 nats**. | This is P3's precondition, stated separately so it can fail on its own. If the offsets disagree, Jev is not a conditional logit with a per-call offset and no amount of anchoring fixes the merge. | |
| **P5** | Because of P1, **`final` beats `flat`** on nDCG@10 at K=240 — the workaround for the cap scores better than the thing it works around. | Four calls of 60 spend 4× the probability resolution on the same candidate set. | |

## Hazards carried in from the measured fingerprint

These are why the strategies are shaped the way they are. All **[MEASURED]** in `systemone`,
none of them documented by the vendor.

- **Probabilities arrive at 2 decimals.** `log p` is undefined at the floor and meaningless one
  step above it. Every logit here is floored at half a step and the floored fraction is reported.
- **Cross-option interaction is real.** IIA fails: log-odds between a fixed pair move
  **+0.31…+0.50** as hard distractors enter the set, CIs excluding zero. A candidate's score is
  a property of its chunk, not of itself — which is the whole reason this repo exists.
- **Below ~0.40 confidence the answer is not reproducible** (same-choice 0.861 in [0.20,0.40),
  1.000 in every bin ≥0.60). Wide reranking lives in exactly that band.
- **`jev-latest` is a moving alias.** Pinned to `jev-1.13.0` here and recorded in every saved row.

---

## Added after the first run, before the run that tests it

The first run exposed a defect in `anchored`, not in the idea behind it. Anchors were taken
as `keys[::step]`, and since `keys` arrives in first-stage order, **anchor #1 is the BM25
top-1 candidate** — a likely winner placed into *every* chunk, where it takes most of the
mass and pushes the rest of that chunk to the 0.00 floor. `anchored` floored **0.914** of the
field against `naive`'s **0.687** at nearly the same chunk width. It matched `flat` anyway,
which is a point in favour of the equating, but the anchor set is chosen badly.

An anchor should be an item that *links* scales, not one that *wins*. Psychometrics says the
same thing: an anchor set should be representative of the scale's range, not of its ceiling.

| # | prediction | outcome |
|---|---|---|
| **P6** | Drawing anchors from first-stage ranks 20+ instead of from rank 0 cuts the floored fraction from 0.914 to below 0.75, and improves `anchored` nDCG@10 by **≥ 0.005** over the rank-0 anchor set. | |

Frozen 2026-09-21, after `runs/scifact-20260921T123132` and before the run that scores it.
