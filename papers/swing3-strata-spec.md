# Swing 3 spec — choose strata by size, not by annotation quality

**Status:** spec, pre-registration. No results here. Owner: the frame-selection line.

The plan was to re-score published frame-selection methods inside
evidence-defined strata, with a churn floor per stratum. Before building it I
computed whether the strata can carry the test. Mostly they cannot, and that
negative is the paper's opening result rather than a scoping note.

## The gate

A per-stratum churn floor inherits the stratum's *n* twice: the floor is
estimated there, and the comparison it corrects is tested there. So the question
is not "does this benchmark annotate evidence?" but "does the stratum hold enough
items to resolve the effect you expect?"

Required *n* for paired McNemar to reach χ²>3.84, computed from each contrast's
**own observed discordance** rather than an assumed rate
(`n = 3.84·d/Δ²`, d = discordance fraction, Δ = accuracy delta). LongVideoBench
long split, Qwen2.5-VL-7B, budget 16:

| contrast | stratum | n available | disc. | Δ | χ² | **n needed** |
|---|---|---|---|---|---|---|
| transcript vs video-only | multi (≥3) | 160 | 23.8% | +10.0 | 5.92 | **91** |
| transcript vs shuffled | single (1) | 439 | 21.9% | −10.5 | 21.09 | **76** |
| BOLT vs uniform-16 | single (1) | 439 | 17.8% | −3.6 | 2.88 | **514** |
| BOLT vs uniform-16 | multi (≥3) | 160 | 15.6% | −3.1 | 0.64 | **614** |
| SigLIP top-16 vs uniform-16 | single (1) | 439 | 21.9% | −1.8 | 0.51 | **2 529** |
| SigLIP top-16 vs uniform-16 | multi (≥3) | 160 | 18.1% | +0.6 | 0.00 | **17 818** |

Read the last column against the third. **Published selection methods differ by
1–3 points, which needs 500–18 000 items per stratum. LVB supplies 160 and 439.**

The contrast inside the table is the finding: the transcript effects need 76–91
and clear comfortably; the selection effects need 500+ and do not come close. Big
effects survive stratification and the 1–3 point effects this audit targets do
not. That is a property of the field's instruments, not of any one paper — and it
is the honest explanation for why our own evidence-count inversion was only ever
significant as a pattern across three cuts and never inside a single cell.

## Suite, chosen by that gate

| benchmark | multi-evidence stratum | verdict |
|---|---|---|
| **HERBench** | 26 806 Q, *every* question ≥3 cues | clears 614 with room. Primary. |
| **CG-Bench** | 12 129 Q, hour-class, annotated clue intervals | clears. Has an official clue-level MCQ arm, so the window experiment is standard protocol rather than something we invent and then defend. Primary. |
| LongVideoBench | 160 | **cannot resolve method differences within stratum.** Keep as the comparability anchor; state the limitation wherever it appears. |
| MUSIC-AVQA v2.0 | ~54 K Q, per-question modality labels | mechanism testbed only: 60 s videos, and the answer format is not 5-way MCQ so letter-logit scoring does not port unchanged. |

## What is actually rerunnable

Narrower than the field looks.

- **Rerunnable (8):** uniform, SigLIP top-*k*, AKS, BOLT-ITS, k-DPP, AdaRD-key,
  cDPP — our reimplementations under one protocol — plus **ViaRL**, the only
  trained selector with released weights.
- **Paper numbers only (2):** Frame-Voyager (no code or checkpoints published),
  K-frames (nothing released; its data depends on annotations we cannot
  regenerate). These get cited, never compared.
- **Unverified:** HiMu. Checking availability.

## Pre-registration

Fixed before any run, so resolvable contrasts cannot be chosen after seeing
results.

1. **In-reach contrasts** are those whose required *n* (computed as above, from
   the anchor model's discordance) is at most the stratum size. Everything else
   is reported as *underpowered*, with its required *n*, and never as a null.
2. **Two model anchors, both reported.** Qwen2.5-VL-7B and a current model. This
   is not optional: on our own data the selection market measured as uniform
   b16→b64 is +2.77 on Qwen2.5-VL-7B and **+5.33 on Qwen3-VL-8B** over the same
   976 items. A single-generation result is a result about that checkpoint.
3. **Floors are per stratum and per model**, from random draws at the same
   budget, never transferred across either.
4. **Frame pathway is held fixed within every contrast.** Arms inherited from
   earlier experiments must have their *video-only* controls diffed first — we
   lost a mechanism claim to a 5.94-point baseline gap between two pathways that
   was invisible in the deltas.
5. Report **exceedance counts, not z-scores**, for floor comparisons: the pairs
   share runs and are not independent.

## Cost, stated honestly

CG-Bench and MUSIC-AVQA need harness *ports*, not config changes — different
schemas, and MUSIC-AVQA's answer format breaks letter-logit scoring outright.
Plus per-benchmark, per-model random draws for the floors. That is the bulk of
the work, which is why this spec is up before any of it is built.
