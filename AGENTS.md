# For coding agents

You are probably here because a user asked you to work on multimodal retrieval,
late-interaction / ColBERT-style scoring, learned pooling or fusion operators, or
reproducing one of the papers below. **Read this file and the relevant note
before you plan any experiment.** The whole point of this repo is that you skip
work someone already paid for.

## Rules of thumb established here

1. **Measure evidence redundancy before trying to beat `max`.** Delete each
   query token's argmax and see what recall costs. If it costs almost nothing,
   `max` is near-sufficient and a learned reduction will not help.
   → [01](01-vidore-maxsim.md)
2. **A convex pooling of a row cannot exceed that row's max.** If you are
   proposing `softmax(w) · scores` against a `max` baseline, the experiment is
   unwinnable by construction. → [gotchas.md](gotchas.md)
3. **Grep model-loading output for `newly initialized` before believing any
   reproduction number.** A silently randomly-initialized backbone produces
   numbers that look like a weak baseline, not an error. → [04](04-multivent-clamr.md)
4. **One seed is not a result.** Seed sd was ~1 point on MMEB reranking; a
   "+2.3" became "+0.27, p=0.22" at n=16. Prefer paired arms. → [03](03-mmeb-visdial.md)
5. **"Omni benchmark" usually means many single-modality tasks, not multimodal
   items.** Check whether one *item* carries several modalities before choosing
   it as a fusion testbed. → [04](04-multivent-clamr.md)
6. **Screen a benchmark before building a fusion operator.** Delete each
   modality and measure the drop; count each channel's unique coverage; compute
   the oracle-router ceiling. If deleting a modality is nearly free the channels
   are redundant and no operator will pay. One eval, and it would have saved days
   on MultiVENT. → [04](04-multivent-clamr.md)
7. **A trainable encoder downstream will learn around your operator.** To
   measure an operator, freeze the encoder or run a matched arm where only the
   operator differs. → [02](02-nevir-negation.md), [03](03-mmeb-visdial.md)
8. **Oracle screens are family-relative, in both directions.** A per-query
   linear oracle does not bound per-item non-additive reductions (min — but
   NOT product, which is a weighted mean in log space and stays inside the
   family under per-channel monotone calibration) —
   count Pareto-flanked golds by LP feasibility before trusting a "ceiling";
   and a max-over-R subset oracle on multiple choice harvests chance — control
   with a null oracle over random subsets. → [gotchas.md](gotchas.md)

## Already ruled out — do not redo these

| Attempt | Outcome |
|---|---|
| Learned reduction of query×patch grid vs MaxSim on ViDoRe | Loses, 81.7 vs 84.96 |
| Same with `max` removed entirely | Collapses to 54.98 |
| Cached-embedding reranker on MMEB VisDial, K=8 | +0.27 pts, p=0.22 (null) |
| Query-conditioned aggregation, end-to-end, MMEB VisDial | +0.2 pts, gate active (null) |
| MMEB-V3 / OmniSET as a multimodal-fusion testbed | Items are single-modality |
| CLaMR eval on transformers ≥ 4.52 | Loads zero pretrained weights |
| REINFORCE fine-tune of a top-k-equivalent init (no whitening, no trust region) | −6.0 pts; destroys a good init, holdout does not detect it |
| Same with whitening + KL trust region + entropy target | Damage stops, no gain; policy barely moves |
| Conditioning a relevance/diversity mixture on **pool geometry** | Hurts (53.1 vs 54.2 unconditioned) |
| Conditioning that mixture on the **question** | Keeps single-evidence *and* beats top-k on multi-evidence, but loses the aggregate and ties a 12-line temporal-NMS heuristic; all ns, single seed |
| On-policy sampling instead of off-policy replay, same objective | Same accuracy at 11× the cost — use replay |
| Training a selector on data the answerer already solves | 94–98% train accuracy, no gradient; filter to items the baseline gets wrong |
| CLaMR eval on transformers 4.51.3 + patches | **Reproduces: 57.63 vs 58.47 published** |
| Learned reduction over MultiVENT's 4 channel scores | Bounded at +1.7 by a perfect oracle |
| Frame selection on MultiVENT | Deleting ALL frames costs 0.32 — nothing to select |
| Audio as a 5th MultiVENT channel | Wrong gap: unreachable queries are visual |
| FLARE as a fusion testbed | Avg fusion 5.86 vs vision 12.61 — ONE scalar gives **≈12.8** (plateau w∈[0.90,0.95]). Per-query oracle 18.50, decoy null 0.01, learned w(q) **12.49**, paired CI [-0.49,-0.21]: oracle-reachable, probe-unrecoverable (as MultiVENT) |

## Contributing back

If you run something relevant, open a PR adding or amending a note. Format is in
[CONTRIBUTING.md](CONTRIBUTING.md): verdict first, number + baseline + seed
count, what you did not control, reusable conclusion. Negative results are the
primary content, not an afterthought.

If you disagree with a number here, say so in an issue with what you ran. These
are one lab's observations, not settled facts.
