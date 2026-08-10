# 07 — Doc-side token compression on CLaMR/MultiVENT 2.0: free to 14×, contested at K=8–16

**Verdict: CLaMR's document token set compresses 14× for free (k-means-mean,
K=64: 57.94 vs full 57.67), and the contested regime for any learned pooling
operator is K ∈ {8, 16}, where cluster-mean loses 2.5–3.5 points but random
selection collapses.** One seed, one GPU-hour, no training — this is the screen
that gates operator work, not a result about operators.

Setup: tokens_combined dump (the deployed 57.63 representation; note 04 recipe),
official scorer (per-channel MaxSim → max over channels), official metric
(torchmetrics nDCG@10, binary diagonal targets — see gotchas on ties). Poolers
pool doc tokens only; pooled vectors L2-renormalized; K split across channels
proportional to token count (within) or unconstrained (across, which forces the
union scorer since pooled tokens lose channel identity).

nDCG@10, full-token reference **57.67** (official run: 57.63):

| K | within k-means | within chunk | within random | across k-means (union scorer) |
|---|---|---|---|---|
| 128 | 57.72 | 56.98 | 56.50 | 57.74 |
| 64 | **57.94** | 56.12 | 53.25 | 58.00 |
| 32 | 57.37 | 54.68 | 51.76 | 56.19 |
| 16 | 55.20 | 53.77 | 48.44 | 57.01 |
| 8 | 54.13 | 52.24 | 41.77 | 54.92 |
| 4 | 49.90 | 49.90 | 30.91 | 52.83 |

Readings, in order of reusability:

1. **The bend is at K≈32→16** (−0.3 → −2.5). Mean doc length is 930 valid
   tokens, so the free zone extends to ~14× — far past the ~3–4× published for
   ViDoRe-style token pooling (Clavié 2409.14683). MultiVENT docs are extremely
   redundant.
2. **Structure exists and hand-designed pooling captures only part of it**: at
   K=8, full−kmeans = 3.5 while kmeans−random = 12.4. A learned operator has a
   measured 3.5-point headroom TARGET over cluster-mean at K=8 (2.5 at K=16) —
   a target, not a bound: renormalized pooling can exceed full-token scores,
   as the K=64 row does (+0.27); K=4 vs
   K=8 differ by 4.2, so the token sets are not unimodal (the kill condition
   for aggregation research fails to fire — the line proceeds).
3. **Modes do not align with channel boundaries at aggressive K**: across-channel
   pooling beats within-channel at every K ≤ 16 (57.01 vs 55.20 at K=16, 54.92
   vs 54.13 at K=8, 52.83 vs 49.90 at K=4) despite the within arm using the
   official scorer. The scorer main effect is measured at
   full tokens: union 57.54 vs official 57.67 (−0.13) — tiny and the wrong sign
   to explain across−within (+1.8/+0.8/+2.9 at K=16/8/4), so pooling freedom,
   not the scorer, drives it (the full-K scorer effect need not transfer
   exactly to pooled K, but it bounds the first-order term).
   One-summary-per-modality is the wrong prior at small K.
4. Pooling perturbs the tie structure (see gotchas): K=64 beating full by +0.27
   is within tie-reshuffle noise until a paired test says otherwise.

Seed sensitivity, measured (3 k-means seeds at fixed K, within-channel):
K=8 = 54.13/53.45/53.47 (mean 53.68, spread 0.68) and K=16 = 55.20/55.30/54.89
(mean 55.13, spread 0.41) — index-time noise ≈ ±0.3, well under the 2.5–3.5
headroom targets; use the means as the contested-regime baselines. Still not
controlled: no query-side compression, tokens are jointly contextualized
(per-channel structure is diluted by 28 joint layers; tokens_separate exists
for the clean-substrate version), K=1 not run (within-channel needs K ≥ 4).

Reusable conclusion: on MultiVENT/CLaMR, run operator experiments at K=8 and
K=16 against within-channel cluster-mean (53.68 / 55.13, 3-seed means), report
torchmetrics only, and treat sub-±0.3 deltas as index-seed noise and sub-point
deltas as tie noise until paired.
