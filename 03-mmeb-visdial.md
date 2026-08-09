# 03 — Reranking on MMEB / VisDial, and a single-seed result that evaporated

**Verdict: clean null. Also the best cautionary tale here about seeds.**

Stage-1 retriever: `TIGER-Lab/VLM2Vec-Qwen2VL-2B` on MMEB VisDial (1000-candidate
pools). Reproduced stage-1 hit@1 = **0.6870** before touching anything.

## Part A — the single-seed trap

A reranking head over cached embeddings, scoring stage-1's top-K:

| K | seed 0 (what I nearly reported) | mean ± sd over **16 seeds** | vs stage-1 0.687 |
|---|---|---|---|
| 8 | 0.700 | 0.6897 ± 0.0087 | +0.27 pts, t=1.27, **p=0.22** |
| 50 | 0.710 | 0.6961 ± 0.0116 | +0.91 pts, t=3.12, **p=0.007** |

Seed 0 was a lucky draw in both cases. The K=8 "win" is nothing. The K=50 result
survives, but at a quarter of the effect size the first seed suggested.

**Seed sd here is ~1 point.** Any reranking claim on this benchmark below ~1
point with fewer than ~10 seeds is noise. I would guess a lot of published
reranking deltas are in this regime.

Note also the shortlist was never the constraint: recall@8 = 0.936 and
recall@50 = 0.993, versus ~0.69 achieved. The head simply was not discriminating.

## Part B — end-to-end, with a proper control

The real question: does **query-conditioned aggregation** help — where the query
picks which of the candidate's own tokens get pooled, rather than mixing the two
representations?

```
alpha_j = softmax_j( q · c_j / tau )      weights over the candidate's own tokens
c_tilde = Σ_j alpha_j c_j                 still in the candidate's space
score   = q · c_pool + gate · FGA(q, c)   gate = 0 at init
```

Two arms, identical seeds / steps / shortlists / encoder path, differing **only**
in the scoring function:

| | seed 0 | seed 1 | mean |
|---|---|---|---|
| `dot` — LoRA fine-tuning only | 0.7280 | 0.7350 | 0.7315 |
| `cond` — LoRA + conditioned aggregation | 0.7300 | 0.7370 | 0.7335 |

**LoRA fine-tuning: +4.5 points. Conditioned aggregation on top: +0.2 points.**

Crucially the **gate was active** (+0.33 and +0.41), so the mechanism was used
and simply did not pay. Across all 8 paired checkpoints the mean difference is
−0.2, so even the sign is unstable.

## Why the two-arm design instead of before/after

Re-encoding candidates is only ~0.99 cosine to the cached stage-1 vectors (bf16
plus different batch padding), which flips ~10% of near-ties. So "must equal
stage-1 at gate=0" is not a reachable bar. Both arms pay that same tax, and it
cancels in the difference. A before/after on one run cannot separate "the LoRA
learned the task" from "the operator helped".

## Reusable conclusion

VisDial image retrieval gives a fusion operator **one utility to marginalize
over** — a single image, a single text query. That is the redundant-evidence
regime of [01](01-vidore-maxsim.md) again. A benchmark that *covers* many
modalities across separate tasks is not the same as a benchmark whose *items*
carry several modalities at once. See [04](04-multivent-clamr.md) for what a
genuine multi-utility item looks like.

## Bugs worth not repeating

- **Training pools must be built the way eval pools are built.** Forcing the
  gold candidate into the shortlist when stage-1 missed it teaches the head to
  distrust stage-1's ordering, and flattens the whole sweep. Resample to rows
  where stage-1 genuinely retrieved the gold instead.
- **Duplicate gold in training pools** — dedupe before insertion.
- Encoder-side traps that produce plausible wrong numbers are in
  [gotchas.md](gotchas.md).
