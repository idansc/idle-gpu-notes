# 01 — Can a learned reduction beat MaxSim on ViDoRe?

**Verdict: no, and the reason is structural. Do not retry this on page retrieval.**

## Setup

Late-interaction retrieval (ColBERT / ColPali style) scores a query against a
document by building a similarity grid `S[q, d]` between query tokens and
document patches, then reducing it:

```
MaxSim:  score = Σ_q max_d S[q, d]
```

The question: replace that hand-designed reduction with a learned one — a
factor-graph-style operator with unary, self and pairwise potentials that learns
*how* to marginalize the grid instead of taking a max.

## What happened

| Reduction | nDCG@5 |
|---|---|
| MaxSim, jointly trained | **84.96** |
| best learned reduction | 81.7 |
| learned, max removed entirely | 54.98 |

Every learned variant lost. Removing `max` from the operator entirely was
catastrophic.

## Why — the part worth keeping

Two measurements explain it, and they generalize:

1. **Deleting every token's argmax patch costs only 0.7 recall.** The evidence
   is not concentrated in the winning patch; it is smeared across many patches
   that all say the same thing.
2. **49.6% of query tokens share an argmax patch** with another query token.

So the grid is highly redundant. When many entries carry the same signal, `max`
is a near-sufficient statistic, and a learned pooling has nothing left to
recover. Worse, there is a hard bound:

> **Any convex pooling of a row is bounded above by that row's max.**

If your learned operator is `softmax(logits) · S`, it *cannot* beat MaxSim. It
can only approach it. Several sweeps looked "flat" for exactly this reason
before the bound was noticed. If you want to beat max, the residual must come
from an unbounded family.

## Traps hit along the way

- **Saturated-softmax-imitates-max kills gradients.** Initializing a softmax
  sharp enough to mimic `max` gives ~zero gradient everywhere. Use a real `max`
  plus a zero-gated residual instead.
- **Double-zero deadlock.** Zero-initializing *both* the gate and the residual's
  output layer means both sides have zero gradient forever. Zero exactly one
  side (keep the gate at zero, scale the other down).
- **Cosine-scale loss saturation.** Contrastive loss over cosine similarities in
  [-1, 1] saturates at ≈ ln(N) with no gradient. Add a CLIP-style learnable
  logit scale (init ~20).
- **Missing 1/√d in multi-head attention logits** produced *exactly* uniform
  softmax over deep projections. Looks like "the model learned nothing"; it is
  actually a scaling bug.

## Reusable conclusion

Before trying to beat a hand-designed reduction, **measure the redundancy of
your evidence first**. Delete the argmax and see what it costs. If it costs
almost nothing, stop — you are in the regime where max wins, and the operator is
not your bottleneck.
