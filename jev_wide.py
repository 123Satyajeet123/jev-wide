"""Pick one, or rank, from more candidates than Jev can look at in a single call.

Jev returns a probability per option, but only for the options in *that* call, and a call
holds a bounded number of tokens. Rank a thousand candidates and you are merging numbers
from different calls -- and a probability is normalised inside its own call, so the merge
is the whole problem. A chunk of fifty weak candidates hands its best one a large share;
a chunk holding three strong ones splits mass three ways. Sorting those together sorts
nothing.

This module is four ways to do that merge and one honest statement of what each costs.
Stdlib only, one file, no client library: drop it in.

The two limits, both measured against `jev-1.13.0` (`MEASUREMENTS.md`):

  255 options   documented by the vendor.
  32,768 tokens undocumented, and for anything longer than a sentence it binds first.
                600-character passages hit the token wall at ~220 options. A 400xx
                `max_tokens_exceeded` is the only signal you get, so `ask` treats it as a
                split-and-retry rather than an error, and the packing estimate never has
                to be right.

Probabilities arrive at two decimal places, so most of a wide field comes back at exactly
0.00 and `log p` is undefined there. Every logit here is floored at half a step and the
floored fraction is reported rather than hidden -- at K=200 it is most of the field, which
is the finding, not a footnote.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"        # pinned: `jev-latest` is a moving alias and thresholds drift under it
OPTION_CAP = 255            # documented
TOKEN_BUDGET = 32_768       # measured; see module docstring
CHARS_PER_TOKEN = 3.2       # deliberately low, so packing errs small and the retry is rare
PROB_STEP = 0.01            # the API's reported precision
LOG_FLOOR = math.log(PROB_STEP / 2)


@dataclass
class Ranking:
    """A ranking, plus what it cost and how much of it was decided by ties."""
    scores: dict[str, float]
    calls: int = 0
    input_tokens: int = 0
    floored: float = 0.0                      # fraction of candidates at the quantisation floor
    anchor_spread: list[float] = field(default_factory=list)   # SD of per-anchor offsets, per chunk

    def top(self, n: int | None = None) -> list[str]:
        order = sorted(self.scores, key=lambda c: -self.scores[c])
        return order[:n] if n else order


# ---------------------------------------------------------------- the one API call

def ask(state: str, instructions: str, items: dict[str, str], model: str = MODEL,
        timeout: int = 180, attempts: int = 7) -> tuple[dict[str, float], int]:
    """One `choice` call. Returns {candidate_id: probability} and the input tokens spent.

    Ids are mapped to `o0..oN` because criteria keys are part of the prompt, and a caller's
    ids are usually noise the model should not be reading.
    """
    keys = list(items)
    criteria = {f"o{i}": items[k] for i, k in enumerate(keys)}
    body = json.dumps({"state": state, "model": model, "questions": {
        "pick": {"type": "choice", "instructions": instructions, "criteria": criteria}}}).encode()
    request = urllib.request.Request(URL, body, {
        "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
        "Content-Type": "application/json"})
    last = attempts - 1
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                answer = json.load(reply)
            pick = answer["answers"]["pick"]
            probabilities = {keys[int(o[1:])]: p for o, p in pick["probabilities"].items()}
            return probabilities, answer.get("usage", {}).get("input_tokens", 0)
        except urllib.error.HTTPError as e:
            if e.code == 400 and b"max_tokens" in e.read():
                raise TooWide(len(items))
            if e.code not in (408, 409, 425, 429, 500, 502, 503, 529) or attempt == last:
                raise
            _backoff(attempt)
        except Exception:
            # Anything else that reached us is a transport failure, and the list of those is
            # not knowable in advance: a run of 4,200 calls turned up `RemoteDisconnected`,
            # which is not a `URLError` and so escaped a clause that named its siblings.
            # Retry on the category, not on an enumeration that the next outage extends.
            if attempt == last:
                raise
            _backoff(attempt)
    raise RuntimeError("unreachable")


def _backoff(attempt: int) -> None:
    """Exponential, with jitter -- eight parallel workers retrying in lockstep is the thing
    that turns one 529 into a thundering herd and the next 529."""
    time.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))


class TooWide(Exception):
    """The request exceeded the token budget. Split the chunk and try again."""


def _ask_or_split(state: str, instructions: str, items: dict[str, str],
                  model: str) -> tuple[list[dict[str, float]], int, int]:
    """`ask`, but a chunk that turns out too wide is halved rather than lost.

    Returns one probability map per call that actually went out, because two halves are
    two normalisations and pretending they are one is the mistake this module is about.
    """
    try:
        probabilities, tokens = ask(state, instructions, items, model)
        return [probabilities], tokens, 1
    except TooWide:
        if len(items) <= 2:
            raise
        keys = list(items)
        half = len(keys) // 2
        out: list[dict[str, float]] = []
        total_tokens = total_calls = 0
        for part in (keys[:half], keys[half:]):
            maps, tokens, calls = _ask_or_split(
                state, instructions, {k: items[k] for k in part}, model)
            out += maps
            total_tokens += tokens
            total_calls += calls
        return out, total_tokens, total_calls


# ---------------------------------------------------------------- packing

def pack(items: dict[str, str], per_chunk: int | None = None,
         budget: int = TOKEN_BUDGET) -> list[list[str]]:
    """Split candidate ids into chunks that should fit one call.

    Bounded by three things at once -- an explicit chunk size, the 255-option cap, and the
    token budget -- because which one binds depends entirely on how long the candidates are.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0.0
    cap = min(per_chunk or OPTION_CAP, OPTION_CAP)
    for key, text in items.items():
        cost = len(text) / CHARS_PER_TOKEN
        if current and (len(current) >= cap or used + cost > budget):
            chunks.append(current)
            current, used = [], 0.0
        current.append(key)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def _logit(p: float) -> float:
    return math.log(p) if p > PROB_STEP / 2 else LOG_FLOOR


def _floored_fraction(maps: list[dict[str, float]]) -> float:
    values = [p for m in maps for p in m.values()]
    return sum(p <= PROB_STEP / 2 for p in values) / len(values) if values else 0.0


# ---------------------------------------------------------------- the four strategies

def _run_chunks(state: str, instructions: str, items: dict[str, str],
                chunks: list[list[str]], model: str, workers: int
                ) -> tuple[list[dict[str, float]], int, int]:
    def one(chunk: list[str]):
        return _ask_or_split(state, instructions, {k: items[k] for k in chunk}, model)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, chunks))
    maps = [m for maps, _, _ in results for m in maps]
    return maps, sum(t for _, t, _ in results), sum(c for _, _, c in results)


def flat(state: str, instructions: str, items: dict[str, str],
         model: str = MODEL, **_) -> Ranking:
    """One call over everything. The baseline, and only legal under both limits."""
    maps, tokens, calls = _ask_or_split(state, instructions, items, model)
    scores: dict[str, float] = {}
    for m in maps:
        scores.update(m)
    return Ranking(scores, calls, tokens, _floored_fraction(maps))


def naive(state: str, instructions: str, items: dict[str, str], per_chunk: int = 50,
          model: str = MODEL, workers: int = 8, **_) -> Ranking:
    """Chunk, then sort every candidate by its raw within-chunk probability.

    This is what the obvious implementation does and it is the thing being measured, not
    a strategy anyone should choose.
    """
    chunks = pack(items, per_chunk)
    maps, tokens, calls = _run_chunks(state, instructions, items, chunks, model, workers)
    scores: dict[str, float] = {}
    for m in maps:
        scores.update(m)
    return Ranking(scores, calls, tokens, _floored_fraction(maps))


def two_stage(state: str, instructions: str, items: dict[str, str], per_chunk: int = 50,
              survivors: int = 10, model: str = MODEL, workers: int = 8, **_) -> Ranking:
    """Chunk, keep the best few from each, then one final call over the survivors.

    The survivors are scored inside a single call, so their probabilities are finally
    comparable. Everyone else keeps a within-chunk score, ranked strictly below them.
    Costs one extra round-trip, in sequence, which is the part that hurts in an agent loop.
    """
    chunks = pack(items, per_chunk)
    maps, tokens, calls = _run_chunks(state, instructions, items, chunks, model, workers)
    kept = [c for m in maps for c in sorted(m, key=lambda k: -m[k])[:survivors]]
    scores = {c: -1000.0 + p for m in maps for c, p in m.items()}     # the tail, below any survivor
    final, final_tokens, final_calls = _ask_or_split(
        state, instructions, {k: items[k] for k in kept}, model)
    for m in final:
        scores.update(m)
    return Ranking(scores, calls + final_calls, tokens + final_tokens,
                   _floored_fraction(maps))


def anchored(state: str, instructions: str, items: dict[str, str], per_chunk: int = 50,
             anchors: int = 8, anchor_skip: int = 20, model: str = MODEL,
             workers: int = 8, **_) -> Ranking:
    """Put the same few candidates in every chunk and use them to equate the chunks.

    If Jev is a conditional logit (McFadden 1974), one call's log-probabilities are the
    candidates' utilities minus that call's log partition function -- a single constant per
    call. Items appearing in every call identify that constant, so subtracting it puts every
    chunk on one scale. This is common-item equating with mean-mean linking (Kolen & Brennan,
    *Test Equating, Scaling, and Linking*, ch. 6), the oldest trick in psychometrics.

    It costs no extra round-trip, unlike `two_stage` -- every call still goes out in parallel.
    `anchor_spread` reports, per chunk, the SD of the offsets the individual anchors imply.
    If the conditional-logit story is right those agree and the spread is small; if they
    disagree, the offset is not a constant, the equating is unsound, and the number says so.
    """
    keys = list(items)
    if len(keys) <= per_chunk:
        return flat(state, instructions, items, model)
    # An anchor links the scales; it should not win them. Taken from the head of the input
    # order, anchor #1 is your first stage's top candidate, and putting a likely winner in
    # every chunk takes the mass in every chunk and floors everything else -- measured at
    # 0.914 of the field against 0.687 for the same chunks without anchors. `anchor_skip`
    # steps past the contenders; the anchors still span the rest of the range.
    pool = keys[anchor_skip:] if len(keys) > anchor_skip + anchors else keys
    step = max(1, len(pool) // anchors)
    anchor_keys = pool[::step][:anchors]
    rest = [k for k in keys if k not in set(anchor_keys)]
    chunks = [c + anchor_keys for c in pack({k: items[k] for k in rest}, per_chunk)]
    maps, tokens, calls = _run_chunks(state, instructions, items, chunks, model, workers)

    logits = [{c: _logit(p) for c, p in m.items()} for m in maps]
    present = [a for a in anchor_keys if all(a in l for l in logits)]
    reference = {a: sum(l[a] for l in logits) / len(logits) for a in present}

    scores: dict[str, float] = {}
    spread: list[float] = []
    for chunk_logits in logits:
        implied = [chunk_logits[a] - reference[a] for a in present]
        offset = sum(implied) / len(implied) if implied else 0.0
        mean = offset
        spread.append(math.sqrt(sum((x - mean) ** 2 for x in implied) / len(implied))
                      if len(implied) > 1 else 0.0)
        for c, l in chunk_logits.items():
            if c not in present:
                scores[c] = l - offset
    for a in present:
        scores[a] = reference[a]
    return Ranking(scores, calls, tokens, _floored_fraction(maps), spread)


STRATEGIES = {"flat": flat, "naive": naive, "two_stage": two_stage, "anchored": anchored}


def rank(state: str, instructions: str, items: dict[str, str],
         strategy: str = "anchored", **kwargs) -> Ranking:
    """Rank any number of candidates. `items` maps your id to the text Jev should read."""
    return STRATEGIES[strategy](state, instructions, items, **kwargs)


def choose(state: str, instructions: str, items: dict[str, str],
           strategy: str = "anchored", **kwargs) -> str:
    return rank(state, instructions, items, strategy, **kwargs).top(1)[0]


# ---------------------------------------------------------------- self-check

def demo() -> None:
    """Both directions against a synthetic conditional logit: equating must recover the
    true order where sorting raw probabilities must not."""
    import random
    rng = random.Random(0)
    utility = {f"c{i}": rng.gauss(0, 2) for i in range(120)}
    truth = sorted(utility, key=lambda c: -utility[c])

    def oracle(chunk: list[str]) -> dict[str, float]:
        exp = {c: math.exp(utility[c]) for c in chunk}          # softmax within the call
        total = sum(exp.values())
        return {c: round(v / total, 2) for c, v in exp.items()}  # rounded to the API's 2 dp

    # a partition whose chunks differ wildly in total mass -- the case the merge has to survive
    order = truth[:8] + truth[8:]                                 # strong candidates bunched first
    chunks = [order[i:i + 20] for i in range(0, len(order), 20)]
    maps = [oracle(c) for c in chunks]

    raw = {c: p for m in maps for c, p in m.items()}
    naive_order = sorted(raw, key=lambda c: -raw[c])

    logits = [{c: _logit(p) for c, p in m.items()} for m in maps]
    anchors = order[::15][:8]
    with_anchors = [c + [a for a in anchors if a not in c] for c in chunks]
    amaps = [oracle(c) for c in with_anchors]
    alogits = [{c: _logit(p) for c, p in m.items()} for m in amaps]
    reference = {a: sum(l[a] for l in alogits) / len(alogits) for a in anchors}
    equated: dict[str, float] = {}
    for chunk_logits in alogits:
        offset = sum(chunk_logits[a] - reference[a] for a in anchors) / len(anchors)
        for c, l in chunk_logits.items():
            equated.setdefault(c, l - offset)
    anchored_order = sorted(equated, key=lambda c: -equated[c])

    def agree_at_10(got: list[str]) -> float:
        return len(set(got[:10]) & set(truth[:10])) / 10

    assert agree_at_10(anchored_order) >= 0.8, agree_at_10(anchored_order)
    assert agree_at_10(naive_order) < agree_at_10(anchored_order), (
        "the negative direction failed: raw merging did as well as equating, so this "
        "partition does not actually exercise the problem")

    assert _logit(0.0) == LOG_FLOOR and _logit(0.5) < 0
    assert len(pack({f"c{i}": "x" * 100 for i in range(600)}, per_chunk=50)) == 12
    assert all(len(c) <= OPTION_CAP for c in pack({f"c{i}": "x" * 10 for i in range(900)}))
    long = pack({f"c{i}": "x" * 4000 for i in range(200)})
    assert all(sum(4000 for _ in c) / CHARS_PER_TOKEN <= TOKEN_BUDGET for c in long), long

    print(f"jev_wide: ok  (anchored {agree_at_10(anchored_order):.1f} vs "
          f"naive {agree_at_10(naive_order):.1f} overlap with truth at 10)")


if __name__ == "__main__":
    demo()
