"""Score the four merge strategies on BEIR scifact, against the published qrels.

The design exists to make one comparison possible. At 200 candidates a flat call still
fits inside the token budget, so `flat` is available as ground truth -- and every chunked
strategy can be scored against the thing it is a workaround for. Chunks of 50 put the
reduction under 4:1 compression, which is what a 1,000-candidate field faces at a 250-wide
call. Whatever holds here is what you get out at K=1,000, where flat is not an option.

First stage is BM25 (`bm25s`), reported as its own row so no strategy can take credit for
retrieval. Metric is nDCG@10 through `pytrec_eval`, the reference implementation BEIR uses.

Ties are the live issue and they are handled identically for every strategy: probabilities
come back at two decimals, so most of a 200-wide field is exactly 0.00, and how those are
ordered decides the score. Every strategy's output is re-scored as
(its score, then first-stage rank), so ties fall back to BM25 and no strategy is quietly
credited with a tie-break it did not make.

    TYPESAFE_API_KEY=... python bench_scifact.py [n_queries] [per_chunk] [strategies]
"""
from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import json
import random
import sys
import time
from pathlib import Path

import bm25s
import pytrec_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_wide

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "scifact"
RUNS = ROOT / "runs"
N_CANDIDATES = 200          # flat fits: 29,678 input tokens measured at 600-char passages
PASSAGE_CHARS = 600
PRICE_PER_MTOK = 0.042      # docs.typesafe.ai, input only; output is unmetered


def load() -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    corpus, queries = {}, {}
    for line in (DATA / "corpus.jsonl").open():
        d = json.loads(line)
        corpus[d["_id"]] = (d.get("title", "") + ". " + d["text"]).strip()
    for line in (DATA / "queries.jsonl").open():
        d = json.loads(line)
        queries[d["_id"]] = d["text"]
    qrels: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for i, line in enumerate((DATA / "qrels" / "test.tsv").open()):
        if i:
            q, c, s = line.split()
            qrels[q][c] = int(s)
    return corpus, queries, dict(qrels)


def first_stage(corpus: dict[str, str], queries: dict[str, str], qids: list[str],
                depth: int) -> dict[str, list[str]]:
    ids = list(corpus)
    engine = bm25s.BM25()
    engine.index(bm25s.tokenize([corpus[i] for i in ids], stopwords="en",
                                show_progress=False), show_progress=False)
    out = {}
    for qid in qids:
        tokens = bm25s.tokenize(queries[qid], stopwords="en", return_ids=False,
                                show_progress=False)
        scores = engine.get_scores(tokens[0])
        out[qid] = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -scores[i])[:depth]]
    return out


def as_run(order: list[str]) -> dict[str, float]:
    """A ranking as descending scores, so the tie-break is ours and not the scorer's."""
    return {c: float(len(order) - i) for i, c in enumerate(order)}


def break_ties(scores: dict[str, float], candidates: list[str]) -> list[str]:
    """Order by strategy score, then by first-stage rank. Applied to every strategy."""
    rank_of = {c: i for i, c in enumerate(candidates)}
    return sorted(candidates, key=lambda c: (-scores.get(c, -1e9), rank_of[c]))


def ndcg_at_10(run: dict[str, dict[str, float]],
               qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(
        {q: qrels[q] for q in run}, {"ndcg_cut.10"})
    return {q: r["ndcg_cut_10"] for q, r in evaluator.evaluate(run).items()}


def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     rounds: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """CI on the per-query mean difference a - b, resampling queries. The standard IR test."""
    keys = sorted(set(a) & set(b))
    diffs = [a[q] - b[q] for q in keys]
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (sum(diffs) / len(diffs),
            means[int(0.025 * rounds)], means[int(0.975 * rounds)])


def main() -> None:
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    per_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    wanted = sys.argv[3].split(",") if len(sys.argv) > 3 else [
        "flat", "naive", "two_stage", "anchored"]
    corpus, queries, qrels = load()
    qids = sorted(qrels)[:n_queries]
    print(f"scifact: {len(qids)} queries, top-{N_CANDIDATES} BM25, chunks of {per_chunk}")

    pools = first_stage(corpus, queries, qids, N_CANDIDATES)
    recall = sum(len(set(qrels[q]) & set(pools[q])) / len(qrels[q]) for q in qids) / len(qids)
    print(f"first-stage recall@{N_CANDIDATES} = {recall:.3f}\n")

    runs: dict[str, dict[str, dict[str, float]]] = {"bm25": {q: as_run(pools[q]) for q in qids}}
    stats: dict[str, dict] = {}
    rows: list[dict] = []

    for name in wanted:
        started = time.time()
        run, calls, tokens, floored, spread, failed = {}, 0, 0, [], [], 0

        def one_query(qid: str) -> dict:
            items = {c: corpus[c][:PASSAGE_CHARS] for c in pools[qid]}
            instructions = (f"Which passage best supports or refutes this claim: "
                            f"{queries[qid]}")
            try:
                got = jev_wide.rank(queries[qid], instructions, items, strategy=name,
                                    per_chunk=per_chunk, workers=4)
            except Exception as e:                       # a lost query is reported, not filtered
                return {"strategy": name, "qid": qid, "error": repr(e)[:160]}
            order = break_ties(got.scores, pools[qid])
            return {"strategy": name, "qid": qid, "calls": got.calls,
                    "input_tokens": got.input_tokens, "floored": got.floored,
                    "anchor_spread": got.anchor_spread, "order": order}

        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, row in enumerate(pool.map(one_query, qids), 1):
                qid = row["qid"]
                if "error" in row:
                    failed += 1
                    run[qid] = as_run(pools[qid])
                else:
                    run[qid] = as_run(row["order"])
                    calls += row["calls"]
                    tokens += row["input_tokens"]
                    floored.append(row["floored"])
                    spread += row["anchor_spread"]
                    row = {**row, "top10": row.pop("order")[:10]}
                rows.append(row)
                if i % 50 == 0:
                    print(f"  {name:10} {i}/{len(qids)}  {time.time() - started:5.0f}s",
                          flush=True)
        runs[name] = run
        stats[name] = {
            "calls": calls, "input_tokens": tokens,
            "cost_usd": tokens / 1e6 * PRICE_PER_MTOK,
            "floored": sum(floored) / len(floored) if floored else 0.0,
            "anchor_spread": sum(spread) / len(spread) if spread else None,
            "failed_queries": failed,
            "seconds": time.time() - started,
        }
        print(f"  {name} done: {stats[name]}\n", flush=True)

    scored = {name: ndcg_at_10(run, qrels) for name, run in runs.items()}
    print(f"\n{'strategy':12} {'nDCG@10':>8} {'vs flat':>18} {'calls':>7} "
          f"{'$':>7} {'floored':>8} {'anchorSD':>9}")
    summary = {}
    for name in ["bm25"] + wanted:
        mean = sum(scored[name].values()) / len(scored[name])
        if name == "flat" or "flat" not in scored:
            delta = ""
        else:
            d, lo, hi = paired_bootstrap(scored[name], scored["flat"])
            delta = f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]"
        s = stats.get(name, {})
        print(f"{name:12} {mean:8.4f} {delta:>18} {s.get('calls', 0):7d} "
              f"{s.get('cost_usd', 0):7.3f} {s.get('floored', 0):8.3f} "
              f"{(s.get('anchor_spread') or 0):9.3f}")
        summary[name] = {"ndcg@10": mean, **s}

    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = RUNS / f"scifact-{stamp}"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({
        "model": jev_wide.MODEL, "n_queries": len(qids), "per_chunk": per_chunk,
        "n_candidates": N_CANDIDATES, "passage_chars": PASSAGE_CHARS,
        "first_stage_recall": recall, "summary": summary,
        "per_query_ndcg": scored}, indent=2))
    with (out / "rows.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nrows -> {out}")


if __name__ == "__main__":
    main()
