# jev-wide

**Rank or pick from more candidates than Jev can look at in one call — and merge the calls correctly.**

One file, stdlib only, no client library. `curl` it into your project.

---

## The limit you will actually hit is not the documented one

| limit | value | documented? |
|---|---|---|
| options per `choice` | **255** | yes — `docs.typesafe.ai/api` |
| tokens per request | **~32,768** | **no** |

Measured by binary search against `jev-1.13.0`: **32,962** reported `input_tokens` accepted, one step
higher returns `400 {"error_type":"max_tokens_exceeded"}`. The ~194-token gap to 2¹⁵ is consistent
with the fixed request scaffolding.

For 600-character passages that wall arrives at **~220 options**, so the 255-option cap never gets a
chance to apply. For 2,000-character RAG chunks it arrives at about **70** — extrapolated at the
4.35 chars/token measured on this corpus, not measured directly. If you are reranking
anything longer than a sentence, the token budget is your real `K`, and `jev-wide` packs against
both limits instead of the one in the docs.

## Why merging calls is the hard part

A probability comes back normalised **inside its own call**. Split 1,000 candidates into chunks and
a chunk of fifty weak candidates hands its best one a large share of the mass, while a chunk holding
three strong ones splits mass three ways. Those numbers are not on the same scale. Sorting them
together sorts nothing.

This is not hypothetical: IIA fails on Jev. Log-odds between a fixed pair of options move
**+0.31 … +0.50** as the other candidates change, with confidence intervals excluding zero. A
candidate's score is a property of the chunk it landed in, not of the candidate.

## The four strategies

| strategy | what it does | round-trips |
|---|---|---|
| `flat` | one call over everything. Only legal under both limits. | 1 |
| `naive` | chunk, then sort by raw within-chunk probability. **What the obvious implementation does.** | 1 (parallel) |
| `two_stage` | chunk, keep the best few per chunk, one final call over the survivors. | 2 (sequential) |
| `anchored` | put the same few candidates in **every** chunk and use them to put all chunks on one scale. | 1 (parallel) |

`anchored` is **common-item equating with mean–mean linking** — the standard method for putting
separately-administered tests on a common scale (Kolen & Brennan, *Test Equating, Scaling, and
Linking*, ch. 6). If Jev is a conditional logit (McFadden 1974), one call's log-probabilities are
the candidates' utilities minus that call's log partition function — a single constant per call.
Items present in every call identify that constant, so subtracting it makes the chunks comparable.
`Ranking.anchor_spread` reports how well that story holds per chunk; if the individual anchors
disagree about the offset, the equating is unsound and the number says so.

## Use it

```python
import jev_wide

ranking = jev_wide.rank(
    state="Does vitamin D supplementation reduce fracture risk?",
    instructions="Which passage best answers this question?",
    items={doc_id: text for doc_id, text in candidates},   # any number
    strategy="anchored",
)
ranking.top(10)      # best ten ids
ranking.floored      # fraction of candidates returned at the 0.00 quantisation floor
ranking.calls, ranking.input_tokens
```

`choose(...)` returns the single best id. `TYPESAFE_API_KEY` is read from the environment.

```sh
python jev_wide.py     # self-check against a synthetic conditional logit. Free, no key needed.
```

## Two things that will bite you

**Probabilities arrive at two decimal places.** At 200 candidates, **95.8%** of the field comes back
at exactly `0.00` — so `log p` is undefined for most of it and the ranking below the top few is
whatever your tie-break says it is. `jev_wide` floors logits at half a step and reports
`Ranking.floored` rather than hiding it. Decide your tie-break deliberately; the benchmark here
falls back to the first-stage order.

**Below ~0.40 confidence Jev's answer is not reproducible.** Measured here: an identical repeated
call changes **7.3% of the top-10**. That is the floor under every number in this README, and it is
why the results section leads with a repeat control rather than a strategy table.

**Do not chunk in first-stage order if you merge naively.** It is the worst case, not the neutral
one — see the controls. Better: do not merge naively.

## Results

BEIR scifact, 300 test queries, BM25 top-200, chunks of 50, `jev-1.13.0`. nDCG@10, with a
paired bootstrap over queries against the `flat` call. `runs/scifact-20260921T123132`.

| strategy | nDCG@10 | vs `flat` | calls | $ | floored |
|---|---|---|---|---|---|
| BM25 only (no reranking) | 0.6617 | −0.100 [−0.138, −0.063] | 0 | 0.00 | — |
| `flat` (one 200-wide call) | **0.7619** | — | 300 | 0.375 | 0.958 |
| `naive` | 0.6789 | −0.083 [−0.103, −0.063] | 1,188 | 0.382 | 0.687 |
| `two_stage` | **0.7663** | +0.004 [−0.008, +0.017] | 1,500 | 0.463 | 0.687 |
| `anchored` | **0.7606** | −0.001 [−0.018, +0.015] | 1,156 | 0.414 | 0.914 |

**Reranking with Jev is worth +0.100 nDCG@10 over BM25. Merge the chunks the obvious way and
you keep +0.017 of it** — `naive` throws away **83% of what you just paid for**, at the same
token cost. `two_stage` and `anchored` are both statistically indistinguishable from the flat
call they are standing in for, which is the point: above the limits, `flat` is not on the menu.

### The controls, which are the reason to believe any of the above

`runs/controls-20260921T123937`. Same query, same candidates, `naive` three times.

| pass | nDCG@10 | vs A | top-10 overlap with A |
|---|---|---|---|
| A | 0.6838 | — | — |
| B — **identical repeat** | 0.6800 | −0.0038 [−0.0135, +0.0042] | **0.927** |
| shuffled — **same candidates, different partition** | 0.7220 | **+0.0382 [+0.0148, +0.0619]** | **0.407** |

Change nothing and **7.3% of your top-10 moves** — Jev is not deterministic in this band, and a
200-wide field sits almost entirely below the 0.40 confidence where it stops being.

Change *only which chunk each candidate landed in* and **59% of your top-10 moves**, with an
nDCG shift whose interval excludes zero. The partition effect is about **eight times the
model's own noise floor**. Your chunking order is not an implementation detail; under `naive`
it is most of your ranking.

The direction of that shift explains why `naive` collapses. Pack in first-stage order and every
strong candidate lands in chunk 1, where they split the mass and each scores low, while chunk
4's best-of-the-weak scores high. Naive merging is not noisy — it is **systematically
anti-correlated with your first stage**. Shuffling spreads the contenders out, which is why the
shuffle *improves* on A rather than just perturbing it.

### Predictions, scored

Frozen in `PREDICTIONS.md` before the first call. Four of six held; the wrong ones stay.

| # | prediction | outcome |
|---|---|---|
| P1 | ≥90% of a 200-wide field returns p=0.00 | ✅ **0.958** |
| P2 | `naive` loses ≥0.05 nDCG@10 to `two_stage` | ✅ **−0.087** |
| P3 | `anchored` within 0.02 of `two_stage` | ✅ **0.006**, at 23% fewer calls and one fewer sequential round-trip |
| P4 | per-anchor offsets agree, SD < 0.5 nats | ✅ **0.181 nats** — Jev behaves like a conditional logit with a per-call offset |
| P5 | chunking *beats* `flat`, because smaller chunks buy resolution | ❌ **refuted.** `two_stage` is +0.004 [−0.008, +0.017]. It ties `flat`; it does not beat it. Resolution was recovered (floored 0.958 → 0.687) and did not convert into nDCG@10 |
| P6 | anchors drawn past the contenders cut floored <0.75 and gain ≥0.005 | *(pending)* |

### Rows we could not collect, reported rather than filtered

14 of 4,144 strategy-query pairs in the main run failed (3 `naive`, 11 `anchored`) and 9 of
3,564 in the controls, all of them server-side: **7× HTTP 529, 5× 503, 1× `RemoteDisconnected`**.
A failed query falls back to first-stage order, which drags that strategy *toward* BM25 — so
`anchored`'s 0.7606 is understated by roughly 0.004, not flattered.

`RemoteDisconnected` is not a `URLError` subclass, so it escaped a retry clause that named its
siblings. `jev_wide` now retries on the category with jittered backoff rather than on an
enumeration that the next outage extends. These runs were collected before that fix and are
reported as collected; they have not been re-run to make them look better.

## Reproduce

```sh
./reproduce.sh          # ~4,200 calls, ~$1.63 at $0.042 / Mtok
```

BEIR scifact, 300 test queries, BM25 top-200 (`bm25s`), nDCG@10 through `pytrec_eval` — the
reference implementation BEIR uses. Chunks of 50 put the reduction under 4:1 compression, which is
what a 1,000-candidate field faces at a 250-wide call. Every row and every per-query score is saved
under `runs/`.

`PREDICTIONS.md` was written and committed **before the first API call**. It is scored in the
results table above, wrong ones included.

## Credits

The measured facts about Jev used here — the IIA violation, the two-decimal quantisation, the
instability threshold, the scaffolding overhead — come from
[systemone](https://github.com/123Satyajeet123/systemone), an open research project on System One
models. The token limit was measured for this repo.

MIT.
