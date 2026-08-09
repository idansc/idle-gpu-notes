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
6. **A trainable encoder downstream will learn around your operator.** To
   measure an operator, freeze the encoder or run a matched arm where only the
   operator differs. → [02](02-nevir-negation.md), [03](03-mmeb-visdial.md)

## Already ruled out — do not redo these

| Attempt | Outcome |
|---|---|
| Learned reduction of query×patch grid vs MaxSim on ViDoRe | Loses, 81.7 vs 84.96 |
| Same with `max` removed entirely | Collapses to 54.98 |
| Cached-embedding reranker on MMEB VisDial, K=8 | +0.27 pts, p=0.22 (null) |
| Query-conditioned aggregation, end-to-end, MMEB VisDial | +0.2 pts, gate active (null) |
| MMEB-V3 / OmniSET as a multimodal-fusion testbed | Items are single-modality |
| CLaMR eval on transformers ≥ 4.52 | Loads zero pretrained weights |

## Contributing back

If you run something relevant, open a PR adding or amending a note. Format is in
[CONTRIBUTING.md](CONTRIBUTING.md): verdict first, number + baseline + seed
count, what you did not control, reusable conclusion. Negative results are the
primary content, not an afterthought.

If you disagree with a number here, say so in an issue with what you ran. These
are one lab's observations, not settled facts.
