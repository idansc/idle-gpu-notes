# 05 — FLARE: the first benchmark where the modalities are actually non-redundant

**Status: screen complete. The hand-designed operator destroys information here —
the first such case in five benchmarks.**

## The result

Controlled screen: the same 53,580 unified queries throughout, only `media_mode`
varies. ImageBind, gallery 87,697 clips / 399 videos, zero missing records.

| query set | media | R@1 | R@5 | R@10 | MedR | n |
|---|---|---|---|---|---|---|
| unified | **vision** | **12.61** | 29.22 | 38.56 | 23 | 53,580 |
| unified | audio | 0.25 | 0.99 | 1.65 | 3356 | 53,580 |
| unified | **both (avg)** | **5.86** | 15.12 | 21.24 | 95 | 53,580 |
| vision | vision | 31.00 | 54.85 | 64.39 | 4 | 86,350 |
| vision | both (avg) | 10.32 | 22.50 | 29.68 | 51 | 86,350 |
| audio | audio | 0.12 | 0.47 | 0.78 | 8718 | 135,003 |
| audio | both (avg) | 0.97 | 2.74 | 4.08 | 2790 | 135,003 |

**On queries built to require both modalities, `normalize((v+a)/2)` scores 5.86
against 12.61 for vision alone — 2.15× worse.** ⚠️ **But the bar for any learned
operator here is ~12.8, not 5.86** — a single tuned scalar recovers the collapse
(see below). Never quote the 5.86 gap as available headroom. The vision-query control
collapses the same way, 31.00 → 10.32.

Mechanism, exactly as the paper describes but now measured on a fixed query set:
audio alone is 0.25 (chance = 1/87,697 = 0.001%, so ~220× chance — live but very
weak), and averaging that into a 12.61 vision signal halves it. The reduction is
dominated by the weaker component.

Audio-liveness gate passes, so the weakness is the audio-text *embedding*, not
our extraction: 240 wavs across 60 videos, median RMS 0.067, 0.8% silent,
cross-clip spectral cosine −0.004 (p95 0.308).

## The collapse is a missing scalar, not a fusion problem

Sweeping a fixed weight on the cached galleries (no GPU, no re-encoding),
`s(w) = w·(q·v) + (1−w)·(q·a)`, same 53,580 unified queries:

| w | 0.00 | 0.50 | 0.80 | 0.90 | **0.93** | 0.95 | 1.00 |
|---|---|---|---|---|---|---|---|
| R@1 | 0.25 | 5.00 | 11.60 | 12.80 | **12.86** | 12.81 | 12.61 |

**One scalar recovers the whole 2.15× collapse** (5.00 → ~12.8). Audio's genuine
contribution at the optimum is **+0.25 R@1 over vision alone** — small, but a
paired bootstrap puts it at [+0.11, +0.39], so it is real.

Replicated independently: a 5-fold **held-out** fit by a second line gives
α\* = 0.95 and held-out R@1 12.81 — same optimum, same value, so the sweep had
not overfit.

> **Correction, and then a correction to the correction.** Both lines reported
> 0.95 → 12.81 because neither grid contained 0.93. A 101-point grid picks
> w = 0.93 → 12.86 in all 5 folds. Two independent implementations agreeing to
> the decimal did **not** catch this: we shared the blind spot, not the
> arithmetic. **That failure mode is the durable lesson here** — replication is
> only evidence where the two runs could have disagreed.
>
> But "the bar is 12.86" was itself false precision, of exactly the kind it was
> correcting. Paired bootstrap over the same queries: **0.93 − 0.95 = +0.049,
> 95% CI [−0.028, +0.123] — includes zero.** The peak is not resolvable at
> n = 53,580. And the 5 folds agreeing is not 5 votes: 5-fold train sets share
> 3/5 of their data, so those argmaxes are heavily correlated. A bootstrap over
> queries puts the argmax at 0.93 only 79.8% of the time (0.90: 10.7%,
> 0.95: 9.1%).
>
> **The bar is a plateau: any w ∈ [0.90, 0.95] gives ≈ 12.8.** Quote it that
> way. Nothing downstream depends on the second decimal — mean fusion destroys
> ~6.9 points and one scalar gives them back.

So the honest reading of the published ImageBind/LanguageBind collapses (2.5× and
7.4×) is *equal weighting*, not "fusion is hard". A single tuned number fixes
them. And once fixed, the second modality is worth 0.2 points — on the benchmark
built specifically so queries require both.

This is the fifth null for "a learned reduction beats a hand-designed one", but
with a new shape: here the hand-designed operator was **badly chosen** rather
than near-optimal, and the repair is one scalar, not a learned function. Anyone
proposing a learned fusion here must beat ≈12.8, not the 5.86 the literature
reports.

Scope: this is score-level fusion of two single-vector embeddings. A token-level
operator could in principle extract more. A **query-conditioned** one cannot —
that is now measured, below.

## What the averaging operator actually destroys

Before asking what could repair the collapse, measure what causes it. Two
mechanisms are bundled in `normalize((v+a)/2)`, and only one had been estimated:

- **Compression** — the score is `(q·v + q·a) / ‖v+a‖` with
  `‖v+a‖ = √(2 + 2·v·a)`, a *per-document* denominator. Measured
  `cos(v,a) = 0.247 ± 0.120`, so `‖v+a‖` ranges 1.31–1.79.
- **Reordering** — `q·a` enters the numerator, and audio-text is near-noise here
  (audio-only R@1 = 0.25).

The control that separates them kills the audio *signal* while keeping the
audio-induced *normalizer*: `(q·v) / √(2 + 2·v·a)`.

| arm | R@1 |
|---|---|
| vision only, `q·v` | 12.61 |
| **dead audio, compression kept**, `q·v / ‖v+a‖` | **10.65** |
| incumbent, `(q·v + q·a) / ‖v+a‖` | 5.86 |
| audio added, no compression, `(q·v + q·a)/2` | 5.00 |

Harness check: the formula reproduces the measured incumbent to **+0.00** against
the arm's own encoded gallery, so this is describing the real operator.

**The dominant term is the injected noise, not the compression.** Under the
corrected model this arm has a definite prediction: if the per-document
denominators were homogeneous, every score would be divided by the same
constant, ranking would be untouched, and it would return **12.61**. It returns
**10.65**, so denominator *heterogeneity* costs 1.96 of the 6.75-point collapse
and adding `q·a` costs 4.79.

(The 7.98 figure this replaces was never a valid expectation, and the reason is
worth keeping: it derived a single 0.63× factor from the *mean* cos(v,a) — that
is, from the very homogeneity assumption under which the effect on R@1 is
exactly zero. A prediction and its own premise cannot both be right here.)

The decomposition is not additive, and reading it in the other order is what
makes the point sharp: adding audio to an *un*-normalized score costs 7.61
(12.61 → 5.00), and then switching to the per-document normalizer **gives back
0.86** (5.00 → 5.86). So the varying denominator — the "documents are penalised
for disagreeing with themselves" story, including my own earlier framing of it —
is a small term, and on top of an already-audio-polluted score it *helps*.

What survives is simpler and blunter: **the operator's failure is that it adds a
noise channel at equal weight.** That is why one scalar repairs it, and it is why
the repair is a weight rather than a better normalizer.

## The per-query oracle is real, and unreachable

If a learned operator is to beat one number, the best `w` has to differ per
query. With two modalities that is exactly solvable, no sampling: the score is
affine in `w`, so `s_d(w) = c_d + w·m_d`, and gold ranks first iff
`(c_g − c_d) + w·(m_g − m_d) > 0` for every distractor. Each distractor is a
half-line; the feasible set is one interval. Solving it exactly matters — on
MultiVENT I published a "+1.69 hard ceiling" that was measuring my 64-sample
Dirichlet sampler, not the ceiling.

| | R@1 |
|---|---|
| vision only, w = 1.0 | 12.61 |
| best global w, grid-tuned held out | **12.86** |
| **per-query ORACLE w** | **18.50** |
| decoy null (same math, random non-gold target) | **0.01** |

**+5.64 points of genuine per-query headroom**, and it is not the freedom of one
fitted parameter: asking the same question of a random non-gold clip succeeds
0.01% of the time. Exact duplicates of the gold — the trap that made MultiVENT's
LP read 99.7% — occur for 0.01% of queries here, so they change nothing. The
feasible intervals are wide (median width 0.288), so this is not knife-edge
precision either; 69% of them contain 0.95, and the other 31% want a different
`w`.

## But no function of the query finds it

Linear head on the ImageBind query embedding, `w(q) = σ(x·θ + b)`, 5-fold
**video-disjoint** CV (clips of one video are near-duplicates, so a random split
leaks), evaluated on the full 87,697-clip gallery.

Two design choices that decide whether this test is fair, both learned the hard
way in the first attempt:

1. *The objective must be the one you report.* A listwise-CE fit picked w = 0.84
   (worth 12.24) where the top-1 optimum is 0.93 (worth 12.86). Under that proxy
   the conditional arm "beat" the constant by +0.41 — an artifact of handicapping
   the control. Refit both with a top-1 surrogate; the constant arm then lands at
   12.84, i.e. it recovers the grid optimum, which is what says the harness is
   sound.
2. *Initialize the conditional arm AT the best constant.* `θ = 0` reproduces `w0`
   exactly, so the head starts with a free pass and can only move if the query
   says something.

| | R@1 |
|---|---|
| learned constant `w` (held out) | 12.84 |
| **learned `w(q)`** (held out) | **12.49** |
| per-query oracle | 18.50 |

**It moved and got worse.** ‖θ‖ = 7.9, `w(q)` spread 0.63–0.96 (sd 0.042) — the
head did not collapse to its initialization, it fit the training folds and
generalized below the constant it started from. Paired bootstrap on the
per-query difference: **−0.349, 95% CI [−0.485, −0.211]** — excludes zero, so
this is a real loss, not a rounding artifact. Of +5.64 available points, a
query-conditioned weight recovers −0.35.

> **A no-op robustness check, recorded because it looked convincing.** Asked for
> seed variance, I ran 5 seeds and got 12.49 five times, sd 0.00 — and nearly
> reported that as stability. It is not: the fit is full-batch deterministic Adam
> and the seed only perturbs a 1e-3 init the optimiser erases. That is one
> solution found five times. When a "robustness" check returns sd exactly 0.00,
> the run is deterministic and you have measured nothing. The variance that
> matters was over queries, and it is the paired CI above.

### The obvious escape — group the queries — also fails

If `w` were predictable at *coarse* granularity the null would be much weaker, so
this is the version to run before believing it. It costs nothing: `hit_grid[q,i]`
already **is** "gold ranks first at grid `w_i` on the full 87,697-clip gallery",
so assigning a `w` per group and reading held-out hits is the exact metric with
no model in between.

Group by a **lexical proxy for audio-relevance**: does the query text mention
sound (`music`, `voice`, `melody`, `speech`, …)? Visible at inference, no
training. Calling this "query type" would oversell it — see the direction problem
below.

| | R@1 |
|---|---|
| global `w` (held out) | 12.86 |
| per-group `w`, mentions-sound yes/no | 12.88 |
| per-group `w`, sound-word count 0 / 1 / 2+ | 12.86 |
| per-query oracle restricted to the grid | 17.69 |

**+0.01, against an available +4.83 within the same grid.** The structure is
real — the groups *do* select different weights — and it is worthless.

Two honest limits on this. The direction is backwards: queries with **no** sound
word prefer w = 0.90 (*more* audio), sound-mentioning ones prefer 0.93. And the
split is 48,576 vs 5,004, since 90.7% of unified queries mention sound at all.
Both say the lexical proxy is not capturing audio-relevance. So this rules out
*this* grouping, not every grouping — though note a better grouping could only
help, and the ceiling it would be chasing is the +4.83 grid oracle.

One scope error is worth naming, because it is the natural objection: the screen
shows averaging *rescuing* audio-targeted queries (0.12 → 0.97) while *destroying*
vision-targeted ones (31.00 → 10.32), which looks like proof that the best `w`
varies decodably. It is not — those are **different query sets**. Within the
unified set, the only set this oracle claim is about, target type is constant.

So FLARE closes in the same shape as MultiVENT: **oracle-reachable,
probe-unrecoverable.** Two benchmarks, two different reasons for the oracle gap
to exist, one conclusion — the gap is a property of the answer key, not of the
query. That is now the second data point, and it is the one that generalises:
*an oracle ceiling is not a target unless something visible at inference time
predicts it.*

Caveats that bound this: linear probe on a single frozen query embedding, single
seed, score-level only. A token-level operator, or one seeing the candidate as
well as the query, is untested and is the remaining live option. Read it as
"null at probe level".

**Calibration.** The paper's query-based Table 3 has no ImageBind vision-only row
— the single ImageBind row is the *fused* variant at 6.35/16.59/23.09. Our fused
arm on unified queries is 5.86, i.e. −7.7%: acceptable harness agreement. Our
vision-only 31.00 has no published counterpart but brackets sanely inside the
paper's own vision group (CLIP 13.89 < 31.00 < SigLIP2 33.98), consistent with
ImageBind's ViT-H tower.

So the **vision-only vs fused contrast on query-based queries is not in the
paper** — it is new content, and it is what shows the operator is the problem
rather than the encoder.

**Caveat that shaped the design:** the three query sets differ in difficulty
(vision-only scores 31.00 on vision queries but 12.61 on unified queries; audio
scores *higher* on unified queries than on audio queries). Compare only within a
fixed query set. This is why the screen holds the query set fixed instead of
inheriting the paper's cross-table pairing — and why the earlier "19.04 → 7.64"
caveat mattered.

---

*(Original note below; the benchmark-selection reasoning is the transferable
part.)*

## Why this one, after three failures

Three benchmarks failed the fusion question for three different reasons:

| benchmark | why it failed |
|---|---|
| [ViDoRe](01-vidore-maxsim.md) | one modality; evidence redundant across patches |
| [MMEB / VisDial](03-mmeb-visdial.md) | items are single-modality |
| [MultiVENT](04-multivent-clamr.md) | items multimodal, but channels mutually redundant (deleting all frames costs 0.32) |

FLARE ([arXiv 2605.10228](https://arxiv.org/abs/2605.10228), data
`YqjMartin/FLARE` CC BY 4.0 ungated, code MIT) is the first with the property the
question needs: **queries carry a hard bimodal constraint by construction.**

A real unified query from `clip-query-unified.jsonl`:

> *"hands making a leather wallet with a watch visible on the wrist, **and a sad
> violin melody in the audio**"*

Visual constraint AND acoustic constraint in one query. Neither modality alone
satisfies it. Contrast MMEB's OmniSET, which renders one meaning four ways in
parallel and never combines them.

Scale: 399 long videos (225.4 h), 87,697 clips, 274,933 queries — 53,580 unified,
86,350 vision-only, 135,003 audio-only. The single-modality query sets are
built-in controls.

## The incumbent operator is provably wrong here

This is what makes it interesting. In `retrieval_adapters.py`, the audiovisual
fusion is, verbatim:

```python
video_feat = normalize(embeddings[ModalityType.VISION])
audio_feat = normalize(embeddings[ModalityType.AUDIO])
feat = normalize((video_feat + audio_feat) / 2.0)
```

An arithmetic mean of two unit vectors. The paper is explicit that this is a
placeholder, not a design: ImageBind and LanguageBind *"provide no official recipe
for combining per-modality embeddings"*, so they use *"the standard late-fusion
practice by average pooling"*.

Reported Text→Clip R@1 (caption-based):

| model | vision-only | naive AV average |
|---|---|---|
| ImageBind | 19.04 | **7.64** |
| LanguageBind | 19.94 | **2.70** |
| Perception AV | 24.55 | 26.48 |
| WAVE-7B | 27.84 | **65.51** |

The authors diagnose it: *"the gulf between vision-only and audio-only embedding
quality [...] is so large that simple pooling is dominated by the weaker audio
component rather than being complemented by it."*

So the same fusion costs ImageBind 2.5×, costs LanguageBind 7.4×, and gains
WAVE 2.4×. That spread is the operator's doing — the opposite of ViDoRe and
MultiVENT, where the hand-designed reduction was near-optimal and unbeatable.

**Caveat we have not yet cleared:** those vision-only and unified numbers may be
computed on *different query sets* (vision queries vs unified queries). If so the
2.5× is partly a population artifact, not a fusion failure. We made exactly this
mistake once already (see the correction in [04](04-multivent-clamr.md)) and will
not inherit the pairing — the harness exposes `media_mode`, so the first run holds
the unified query set fixed and varies only the media scope.

## The ladder to test

1. naive average — the incumbent
2. **best fixed per-modality weight** — a 2-parameter fix. If this recovers most
   of the gap, the "fusion is broken" story is really "nobody tuned a scalar",
   and that is the honest baseline any learned operator must beat.
3. query-conditioned weighting — the only thing that can use audio on *"a sad
   violin melody"* and ignore it on *"text on black screen saying subscribe"*

Step 2 is the one the field skipped by inheriting a placeholder, and skipping it
would let us claim credit for a scalar.

## Practical notes

- 79 GB of videos as 14 zips of ~6,600 clip mp4s, `<video_id>/<video_id>-Scene-NNN.mp4`,
  matching the metadata `video_path` exactly. Metadata is a separate 1 GB.
- **Audio does not ship as audio.** It is inside the mp4s; every audio encoder in
  the harness wants wav paths, so 87,697 ffmpeg extractions are unavoidable.
- **No train split.** The paper evaluates pretrained retrievers zero-shot, so any
  trained result is on a split you invent — and it must be split by the 399 source
  videos, never by clip, or clips of the same video leak across the boundary.
- Harness natively supports three modality scopes (vision / audio / unified) and
  two query regimes (caption-based / query-based), so the screen below is free.

## The screen, before building anything

The three diagnostics that would have saved us on MultiVENT, plus a fourth for
the conjunctive structure this benchmark claims, run first this time:

1. **marginal contribution** — delete a modality, measure the drop
2. **unique coverage** — queries this channel retrieves that nothing else does
3. **oracle ceiling** — per query, the best channel or weighting; the bound on
   per-query *linear* operators only (gotchas.md)
4. **Pareto-flanking count** — golds no linear weighting can rank first (LP
   feasibility, one constraint per distractor) that `min` still ranks first: the
   operator-free measure of score-level conjunctiveness. The diagnostic
   reports TWO counts, because a material flanked count is *necessary* for
   calibrated-min (per-channel affine, then min) to beat the best fixed
   weighted mean, not sufficient: min's failure mode is brittleness, tanking
   golds whose weakest calibrated channel sits within calibration noise of the
   distractor field. Pre-registered prediction: calibrated-min wins iff
   flanked-gold gains minus min-vulnerable losses clear zero. Calibration run
   on MultiVENT's tensor (done, CPU): flanked golds are 11.3% (170/1504) — the
   "~0 expected" guess was wrong — but the instrument is valid (decoy null:
   0.8% of random non-golds are linearly rankable-first vs 90% of golds), and
   the NET is what note 04 predicts: min-rescued 8 vs min-vulnerable 503
   (z-min nDCG@10 = 33.8 vs max 57.4; even on the flanked subset z-min puts
   4.1% in top-10 vs max's 6.5%). Flanked-but-unrescuable means flanking alone
   is not conjunctiveness — on FLARE, read only the net.
   Binds on the calibrated-min arm, so the comparison is one-variable: both
   arms score the SAME per-channel affine-calibrated inputs (weighted mean of
   calibrated vs min of calibrated); the calibration is fit on a video-level
   holdout (split by the 399 source videos, never by clip); and the gap is
   pre-registered to appear on unified queries only, with vision-only queries
   as the ~0-gap control

If audio's marginal contribution on unified queries is large, this benchmark can
support the fusion question and the operator work is justified. If it is ~0.3 like
MultiVENT's frames, we stop and write up four nulls instead of three.

## Screen results (2026-08-12) — the branch fires

Grid complete: 7 configs, fixed-query-set / varied-media-scope, ImageBind,
87,697-clip gallery; replicated cross-hardware (mm-lab01 mirror matches the
cluster to 0.003 R@10 on the audio arm). Gate 2 closed: our fused 5.86 vs the
paper's published fused 6.35 (−7.7% harness agreement); our vision-only 31.00
on vision queries brackets between the paper's CLIP (13.89) and SigLIP2
(33.98). The controlled vision-vs-fused contrast below is NEW — the paper
never ran it query-based.

Text→media R@1 / R@10 / MedianRank, unified queries (n=53,580) unless noted:

| arm | R@1 | R@10 | MedR |
|---|---|---|---|
| vision-only | 12.61 | 38.57 | 23 |
| incumbent fusion (normalized-vector mean) | 5.86 | 21.25 | 95 |
| **best fixed weight, w = 0.93** | **12.86** | — | — |
| audio-only | 0.25 | 1.65 | 3356 |
| score-mean | 5.00 | 18.84 | 122 |
| per-item max | 9.73 | 31.42 | 42 |
| per-item min | 2.09 | 7.86 | 1254 |
| calibrated (z) min | 2.46 | 9.73 | 651 |
| best fixed α, VIDEO-LEVEL HOLDOUT (α grid lacked 0.93) | 12.81 | 39.07 | 22 |

α* = 0.95 on every one of five source-video folds — the tuned scalar is
vision-plus-a-whisper and ties vision-only, exactly the pre-registered branch:
**on ImageBind, the scalar ends the ladder; no operator arms run on this
substrate; operator work moves to the strong-audio incumbents** (WAVE-7B
query-based anchor 42.63 — NOT the caption-based 65.51; Qwen2.5-Omni-7B;
Qwen3-Omni-30B; published vision ceiling for the incumbent table:
Qwen3-VL-Emb-8B 60.82).

The asymmetry that is the mechanism: the same average RESCUES audio queries
(0.12 → 0.97 R@1, 8×, n=135,003) while destroying vision queries (31.00 →
10.32, 3.0×) and the bimodal-constraint queries (12.61 → 5.86, 2.15×) — both
repaired by one scalar to 12.86. One
line of algebra says why: the normalized-mean score is
(q·v + q·a) / sqrt(2 + 2 v·a); measured cos(v,a) = 0.25 ± 0.12 compresses
vision margins by ~0.63× while q·a (sd 0.075, pure noise here) reorders the
compressed near-ties.

Pre-registered instrument outcomes (stated before the numbers landed, in the
fleet log): raw flanked count came out LARGE (81.5%) and uninformative as
registered — with one dead channel and an 87,697 gallery it measures
reachability, not conjunction (only 18.2% of golds are rank-1-able and 45.9%
top-10-able under ANY α; contrast MultiVENT: 89.3% rankable-first in a 1,504
pool with four live channels). Decoy: 99.99% infeasible, trivially. The
load-bearing dead-channel zero held: min-rescued 3,178 vs min-vulnerable
14,799 — net −11,621, and on the flanked subset z-min reaches top-10 for 6.0%
vs max's 19.2%. The conjunctiveness question is UNTESTABLE on this substrate
and moves with the operator work.
