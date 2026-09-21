"""The two controls the strategy table is worthless without.

A difference between merge strategies only means something if it is bigger than the
difference you get from changing nothing. Two things move a wide ranking without any
strategy being involved, and both are measured here rather than assumed away:

  repeat     identical query, identical candidates, identical partition, called again.
             Jev is deterministic above ~0.60 confidence and demonstrably not below
             ~0.40, and a 200-wide field sits almost entirely below 0.40. This is the
             noise floor: no strategy difference smaller than this is real.

  shuffled   same candidates, same chunk size, different partition. Nothing about the
             task changed -- only which rivals each candidate was shown next to. Under
             IIA this would cost nothing. IIA fails on Jev (log-odds between a fixed
             pair move +0.31...+0.50 as distractors change), so this is the size of the
             problem `jev_wide` exists to solve, measured directly.

    TYPESAFE_API_KEY=... python controls_scifact.py [n_queries] [per_chunk]
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jev_wide
from bench_scifact import (DATA, N_CANDIDATES, PASSAGE_CHARS, PRICE_PER_MTOK, RUNS,
                           as_run, break_ties, first_stage, load, ndcg_at_10,
                           paired_bootstrap)


def main() -> None:
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    per_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    corpus, queries, qrels = load()
    qids = sorted(qrels)[:n_queries]
    pools = first_stage(corpus, queries, qids, N_CANDIDATES)
    print(f"controls: {len(qids)} queries, top-{N_CANDIDATES}, chunks of {per_chunk}\n")

    def run_pass(label: str, shuffle_seed: int | None) -> tuple[dict, dict, list[dict]]:
        started = time.time()
        def one(qid: str) -> dict:
            order = list(pools[qid])
            if shuffle_seed is not None:
                random.Random(shuffle_seed).shuffle(order)
            items = {c: corpus[c][:PASSAGE_CHARS] for c in order}
            instructions = (f"Which passage best supports or refutes this claim: "
                            f"{queries[qid]}")
            try:
                got = jev_wide.rank(queries[qid], instructions, items, strategy="naive",
                                    per_chunk=per_chunk, workers=4)
            except Exception as e:
                return {"pass": label, "qid": qid, "error": repr(e)[:160]}
            # tie-break always falls back to the ORIGINAL first-stage order, so a shuffle
            # cannot be credited or blamed for a change in how ties resolve
            return {"pass": label, "qid": qid, "calls": got.calls,
                    "input_tokens": got.input_tokens, "floored": got.floored,
                    "order": break_ties(got.scores, pools[qid])}
        rows, run = [], {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row in pool.map(one, qids):
                rows.append(row)
                run[row["qid"]] = as_run(row.get("order") or pools[row["qid"]])
        tokens = sum(r.get("input_tokens", 0) for r in rows)
        stat = {"calls": sum(r.get("calls", 0) for r in rows), "input_tokens": tokens,
                "cost_usd": tokens / 1e6 * PRICE_PER_MTOK,
                "failed": sum("error" in r for r in rows),
                "seconds": time.time() - started}
        print(f"  {label}: {stat}", flush=True)
        return run, stat, [{k: v for k, v in r.items() if k != "order"} for r in rows]

    passes, stats, rows = {}, {}, []
    for label, seed in (("A", None), ("B", None), ("shuffled", 7)):
        passes[label], stats[label], new = run_pass(label, seed)
        rows += new

    scored = {k: ndcg_at_10(v, qrels) for k, v in passes.items()}
    overlap = {}
    for label in ("B", "shuffled"):
        same = [len(set(passes["A"][q]) & set(passes[label][q])) for q in qids]  # sanity: same pool
        top10 = [len({c for c, s in sorted(passes["A"][q].items(), key=lambda x: -x[1])[:10]}
                     & {c for c, s in sorted(passes[label][q].items(), key=lambda x: -x[1])[:10]})
                 / 10 for q in qids]
        overlap[label] = sum(top10) / len(top10)
        assert all(s == N_CANDIDATES for s in same)

    print(f"\n{'pass':12} {'nDCG@10':>8} {'vs A':>20} {'top10 overlap w/ A':>20}")
    out = {}
    for label in ("A", "B", "shuffled"):
        mean = sum(scored[label].values()) / len(scored[label])
        if label == "A":
            delta, ov = "", ""
        else:
            d, lo, hi = paired_bootstrap(scored[label], scored["A"])
            delta = f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]"
            ov = f"{overlap[label]:.3f}"
        print(f"{label:12} {mean:8.4f} {delta:>20} {ov:>20}")
        out[label] = {"ndcg@10": mean, "delta_vs_A": delta, "top10_overlap_vs_A": ov,
                      **stats[label]}

    RUNS.mkdir(exist_ok=True)
    d = RUNS / f"controls-{time.strftime('%Y%m%dT%H%M%S')}"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(
        {"model": jev_wide.MODEL, "n_queries": len(qids), "per_chunk": per_chunk,
         "passes": out, "per_query_ndcg": scored}, indent=2))
    with (d / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nrows -> {d}")


if __name__ == "__main__":
    main()
