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

**One scalar recovers the whole collapse: 5.86 → ~13.1, +7.2 points, without
leaving the operator the benchmark already uses.** (Plateau, w ≈ 0.87–0.90 — not
a two-decimal argmax. The same curve gave 13.09 at w=0.90 on a coarse grid and
13.12 at w=0.87 on a fine one, and the nested per-fold picks span 0.87–0.89. That
0.03 sits inside noise this benchmark cannot resolve at n = 53,580 — the third
time today the same trap has been laid. The number carrying the claim is the
nested margin, which is selection-free.) Audio's genuine contribution
is **+0.474 R@1 over vision alone**, nested 95% CI [+0.271, +0.664].

> ⚠️ **Family correction — read before quoting any number below.** The incumbent
> is `q · normalize(w·v + (1−w)·a)`. Every sweep in the original version of this
> note was the **linear** family `w·(q·v) + (1−w)·(q·a)`, which agrees with it
> only at w = 0 and w = 1. That made three published numbers wrong:
>
> | | linear (what I swept) | renormalized (the real family) |
> |---|---|---|
> | best fixed w | 0.93 | **≈0.87–0.90** |
> | bar | 12.86 | **≈13.1** |
> | audio margin over vision-only | +0.254 [+0.063, +0.386] | **+0.474 [+0.271, +0.664]** |
>
> The bar is **≈13.1**, not the ~12.8 plateau I circulated. Audio's margin
> roughly **doubles** and moves clearly away from zero, so "significant but
> tiny" is too weak — half a point is small, not negligible. Sections below that
> still quote linear-family numbers are marked.

That range is deliberate. Quoting the *peak's* advantage (+0.25, paired CI
[+0.11, +0.40]) inherits the selection, because w = 0.93 is the argmax of the
same data. The selection-free statement is the plateau's worst endpoint, and it
is markedly more fragile:

| against vision-only (w = 1.0) | Δ R@1 | paired 95% CI |
|---|---|---|
| w = 0.90 | +0.194 | [+0.017, +0.373] |
| w = 0.93 (argmax) | +0.254 | [+0.112, +0.398] |
| w = 0.95 | +0.205 | [+0.088, +0.321] |

Every point on the plateau beats vision-only with a CI excluding zero.

**The headline number is the nested one, on the renormalized family: +0.474,
95% CI [+0.271, +0.664].** (The +0.254 [+0.063, +0.386] below is the same
procedure on the linear family — kept because the method discussion is about the
procedure, not the operator.)
Rather than argue about which fixed `w` to quote, remove the selection: for each
held-out fold, pick `w` on the *other* folds and take the difference against
vision-only inside that fold. No grid-resolution argument, no round-number
defence — and it estimates what a deployment actually faces, "tune the weight on
your data, apply it to new data, versus drop audio".

The bootstrap has to be nested too, or it smuggles the bias back: resampling
queries while reusing weights chosen on the full data holds selection fixed and
understates its variance. Redoing the selection inside every resample:

| | Δ R@1 | 95% CI | width |
|---|---|---|---|
| **nested selection, nested bootstrap** | **+0.254** | **[+0.063, +0.386]** | 0.323 |
| weights fixed, queries resampled | +0.254 | [+0.112, +0.398] | 0.286 |

Two things worth reading off this. The **point estimate is unchanged** — nested
selection picks w = 0.93 in all five folds, so the bias I was worried about was
never in the estimate. It was in the **interval**: the naive bootstrap is 1.13×
too narrow and puts the lower bound at +0.112 where the honest one is **+0.063**.

Robustness at fixed weights: 0.90 → +0.194, 0.93 → +0.254, 0.95 → +0.205.

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

> **Correction to the justification, not the conclusion.** "The repair is a
> weight" survives and is stronger than stated: within the incumbent's *own*
> family, moving w from 0.5 to 0.87 goes 5.86 → 13.12. But I justified it by
> calling the normalizer a small, order-dependent term, and that is wrong. Swept
> across w, renormalizing **helps everywhere audio carries real weight** —
> +0.86 at w=0.5, **+1.68 at w=0.7**, +0.29 at w=0.9, decaying to 0 at w=1 where
> the families coincide. Dropping it costs you. The incumbent's failure was the
> weight alone; its normalizer was never the problem.
>
> The order-dependence I found earlier was real but was a two-point artefact of
> comparing families at a single w. The full curve settles it.

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
| **per-query ORACLE w, continuous** *(linear family)* | **18.50** |
| per-query oracle restricted to the 11-point grid *(linear)* | 17.69 |
| per-query oracle on the renormalized family, 23-point grid | 17.76 |
| decoy null, renormalized family, same grid | 0.01 |
| decoy null (same math, random non-gold target) | **0.01** |

⚠️ **Family note:** the oracle, probe and grouping results in this section were
computed on the **linear** family, so they pair with the 12.86 bar, not 13.12.
The renormalized oracle is 17.76, essentially unchanged, which is why the
qualitative null is expected to hold — the nulls are about whether `w` is
predictable per query, not about the denominator. "Expected" is not "measured",
and re-deriving the probe on the renormalized family is the one loose end left
on this benchmark.

⚠️ Two oracles, and they answer different questions. **18.50** is the geometry
claim: some real `w` exists. **17.69** is the ceiling for anything choosing from
the grid, and it is the one to compare the grid-restricted arms against —
subtracting 12.86 from 18.50 credits those methods with +5.64 when only +4.83 was
ever reachable by them.

**+5.64 points of genuine per-query headroom**, and it is not the freedom of one
fitted parameter: asking the same question of a random non-gold clip succeeds
0.01% of the time. Exact duplicates of the gold — the trap that made MultiVENT's
LP read 99.7% — occur for 0.01% of queries here, so they change nothing. The
feasible intervals are wide (median width 0.288), so this is not knife-edge
precision either; 69% of them contain 0.95, and the other 31% want a different
`w`.

## ⚠️ RETRACTED on the correct family: a function of the query DOES find some of it

**Everything in this section was computed on the linear family.** Re-run on the
renormalized family — the operator the benchmark actually uses — the result
reverses:

| | linear family | **renormalized family** |
|---|---|---|
| learned constant `w` (held out) | 12.84 | 13.08 |
| **learned `w(q)`** (held out) | 12.49 | **13.28** |
| `w(q)` − const, paired | **−0.349** [−0.485, −0.211] | **+0.159** [mean of 5 partitions] |

Same code, same video-disjoint folds, same top-1 surrogate, same
initialize-at-the-best-constant free pass. Only the operator changed, and the
sign of the headline flipped from significantly worse to significantly better.

Two checks that this is a real reversal and not a bug. The held-out constant
lands at 13.08 by two independent routes — this fit's 51-point pool-proxy
selection, and the nested grid run's margin (+0.474 over vision-only 12.61 =
13.08). And `w(q)` at 13.28 also beats the best *fixed* grid point, 13.12, so it
is not merely beating a poorly-chosen constant.

**What it actually recovers: +0.159 of +4.68 available, i.e. 3.4% of the oracle
gap.** So the honest claim is neither the old null nor a win:

> A query-conditioned weight is measurably better than any fixed one, and
> recovers about 4% of the oracle gap.

**But +0.192 is not the finding — the flip is.** Whether a learned operator beats
a fixed one is decided here by a modelling choice almost no paper reports: not
the architecture, not the objective, not the data, but the operator *family* the
comparison is embedded in. And the two families are not arbitrary alternatives —
one is what the benchmark actually uses, the other is the one that is natural to
write down. Two honest groups could run identical code on identical data and
publish opposite conclusions about learned-vs-fixed operators.

The bound makes it sharper still: **+4.68 is available, hand-specified structure
gets ~0% of it (five groupings, none beating a global weight), a learned
per-query head gets 4%, and the remaining 96% is unexplained.** So the structure
is real, it is per-query, and it is not captured by any partition of queries we
could specify by hand.

Why the family decides this — and the version of this sentence I wrote first was
wrong. I said the linear head "had nothing to exploit but noise". That cannot be
right: the linear family has **+4.83** of oracle headroom against the
renormalized family's **+4.68**. Essentially the same structure is present in
both.

What differs is not how much structure exists but **whether it is reachable from
the query**. In the linear family the score is affine in `w`, so a per-query `w`
only slides each document along a line and the reachable axis is one a linear
head on the query embedding cannot predict. Renormalizing makes `w` also move
the *per-document* denominator, and that axis is predictable. Same headroom,
different accessibility.

Stated generally, which is the part that ports:

> **The operator family determines whether existing structure is reachable, not
> whether it exists.**

The denominator doing the work here is the same term this note called "a small,
order-dependent effect" this morning. It now carries the only positive result on
the board.

**What survives unchanged**, all re-derived on the renormalized family: the decoy
null (0.01), and every grouping arm (five of them, none beating a global weight).
So *group*-unrecoverable holds; it is specifically **per-query** conditioning
that works, and only on the family whose denominator varies per document.

---

*(Original linear-family section below, kept because the two design rules in it —
fit with the metric you report, and initialize the conditional arm at the best
constant — are what make either result trustworthy.)*

## The linear-family version: no function of the query finds it

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

**+0.01, against an available +4.83 within the same grid.**

But this proxy is bad evidence on its own, and the reason is visible in its own
output: the direction is backwards — queries with **no** sound word prefer
w = 0.90 (*more* audio) — and the split is 48,576 vs 5,004, since 90.7% of
unified queries mention sound at all. A feature anti-correlated with the thing it
is named for may have failed because it does not measure audio-relevance, not
because audio-relevance is unhelpful.

So the lexical test alone would only license "unrecoverable by *this* proxy".
The general claim needs a grouping not derived from words at all.

### Groupings with no semantics in them

Bin queries by their own retrieval statistics — all visible at inference, no
lexicon, no training. `max_d q·v` is how confidently vision retrieves anything;
`max_d q·a − max_d q·v` is the score-space analogue of "is this query audio-ish".

| grouping (held out, exact) | R@1 | Δ vs one global `w` |
|---|---|---|
| one global `w` | 12.86 | — |
| `max q·v`, 2 / 5 / 10 bins | 12.86 / 12.82 / 12.75 | +0.00 / −0.04 / −0.12 |
| `max q·a`, 5 bins | 12.75 | −0.11 |
| audio affinity, 2 / 5 / 10 bins | 12.83 / 12.81 / 12.83 | −0.03 / −0.05 / −0.04 |
| **per-query oracle, restricted to grid** | **17.69** | **+4.83** |

**Seven groupings, not one beats a single global number, and most are worse.**
Every one of them *does* select different weights per bin — the picks span
0.85–1.00 — and finer binning makes held-out performance *degrade*, which is the
signature of a per-group argmax that fits its bin and does not transfer.

That is the shape of the whole result in miniature: the structure is there, and
it does not survive being asked to predict.

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

**Calibration, and a retraction.** This note previously claimed the vision-only
vs fused contrast on query-based queries "is not in the paper — it is new
content". **That was wrong.** Table 4 (T→Clip, query regime) carries both sides
for ImageBind, and our arms reproduce both:

| | published | ours | Δ |
|---|---|---|---|
| ImageBind fused (unified media) | 6.35 | 5.86 | −7.7% |
| ImageBind vision-only | 12.87 | 12.61 | −2.0% |

Two-sided agreement inside 8% is a better harness check than the one-sided one
we had, so the correction strengthens the reproduction while removing the
novelty claim. **The collapse was visible in the published table all along.**
What is new here is the explanation (an equal-weight noise channel), the repair
(one scalar), and the bound on repairing it further (oracle 18.50, probe 12.49,
eight groupings ≤ +0.01) — not the observation.

## Pre-registered: the WAVE contrast

Written before the galleries merged, so the result is a test rather than a
description. Recorded here even if it goes against me.

**The experiment.** WAVE emits one joint vector per media mode, so there is no
incumbent operator to repair and no `w` to sweep — the ImageBind question does
not transfer. What the three arms give instead is WAVE's *own* vision-only and
audio-only embeddings, so: late-fuse those two with a tuned scalar and compare
against WAVE's native joint output.

| | R@1 |
|---|---|
| WAVE vision-only | 16.80 (published) |
| WAVE audio-only | 7.58 (published) |
| WAVE late-fused, tuned `w` | **?** |
| WAVE native joint | 42.63 (published) |

**Prediction, with a number: tuned late fusion lands below 29.7**, i.e. it
recovers less than half the 16.80 → 42.63 gap. If it comes in above that, the
prediction is wrong and the write-up says so rather than accommodating it.

Why it matters either way. Falling well short is evidence that joint encoding
does something score-level combination provably cannot — a stronger operator
claim than anything else in this note, and it lands against published anchors on
both ends rather than against our own numbers. Reaching 42.63 would instead mean
late fusion is fine whenever both channels are alive, and the entire ImageBind
story reduces to one dead channel plus one bad weight.

**Harness gate — and the gate I first wrote was already too loose.** The audio
arm must reproduce the published 7.58 before anything downstream is trusted. I
originally set the band at **7.0–7.6**, widened to accommodate our own
unexplained ImageBind offset (fused −7.7%, vision-only −2.0%, both negative).

The first WAVE audio arm came in at **7.39 — inside that band, and wrong**: a
shared per-clip frame budget was subsampling audio second-offsets to 32s where
the protocol wants 64. A 2–4% same-signed undershoot at every cutoff, which the
band would have passed as "harness validated" and fed straight into the fusion
rows where audio is one of two inputs.

**The lesson is about how tolerances get set.** A gate should be sized by the
expected magnitude of harness noise, not by an offset you already carry and
cannot explain — widening it to cover your own unknown is exactly how a gate
stops being a gate. Corrected rule: the published value plus a stated decode
tolerance, and **any systematic same-sign pattern across cutoffs is a fail
regardless of magnitude**, since that is the signature of truncation rather than
noise.

Our own ImageBind offset remains **unexplained** — but it is not uniform, and
saying so corrects a claim I made here.

**It scales with how much audio is in the score.** FLARE publishes a third
ImageBind anchor we had not compared:

| arm | published | ours | Δ |
|---|---|---|---|
| vision-only | 12.87 | 12.61 | −2.0% |
| fused | 6.35 | 5.86 | −7.7% |
| **audio-only** | **0.31** | **0.25** | **−19.4%** (≈−17% at R@5 and R@10 too) |

So my earlier line — "a uniform offset moves all the arms together, orderings
survive" — is **refuted by our own numbers**. The deficit is audio-correlated,
which is precisely the axis every `w`-dependent quantity varies along. The
direction is favourable: an under-represented audio channel pushes the optimal
`w` *up* and the audio margin *down*, so the true optimum is likely below 0.87
and audio's true contribution likely larger than +0.474. The claims survive and
the numbers are **conservative** — worth stating out loud rather than leaving
implicit.

### The decode-budget explanation: no evidence, but not falsified

The natural account — upstream ImageBind samples `clips_per_video=3 × 2s ≈ 6s`
of audio against `5 × 2s ≈ 10s` of video, so audio is truncated hard and video
barely — predicts that audio-only should sit near its ceiling on short clips and
degrade as clips lengthen. Two tests, no GPU:

**Integrity.** Extracted wav duration versus the clip's own duration, n=600:
median relative difference **+0.0000**, p05 −0.0002, p95 +0.0001, **0.0%**
mismatched by more than 5%, all 16 kHz mono. So the audio corresponds exactly to
the clip being retrieved — no boundary-alignment or span bug.

**Mechanism.** R@1 stratified by gold-clip duration (all 53,580 resolved):

| clip duration | n | audio-only | vision-only |
|---|---|---|---|
| 0–3s | 221 | 0.00% | 9.95% |
| 3–6s | 21,352 | 0.22% | 13.14% |
| 6–10s | 14,473 | 0.26% | 13.03% |
| 10–15s | 7,997 | 0.28% | 12.05% |
| 15–25s | 5,865 | 0.36% | 11.92% |
| >25s | 3,672 | 0.22% | 10.35% |

**Audio-only goes UP past the 6s budget, not down** — 0.22% under 6s versus
0.27% over it, +26% relative, the opposite of the prediction. Vision-only drifts
down 6.3%, consistent with longer clips simply being harder rather than with a
10s budget biting. 60% of clips exceed 6s, so the budget had ample opportunity to
show itself.

⚠️ **State this as "no evidence of a binding budget", not "falsified", because
the profile carries a confound of the opposite sign.** Longer clips contain more
audio content and are intrinsically more retrievable from audio; truncation would
hurt exactly those clips. The observed rise is the *sum* of the two effects, and
they can cancel. What is genuinely ruled out is that truncation **dominates** the
profile — not that it is absent. Do not cite this table as showing decode budgets
do not matter in general.

The confound-free version, if it ever matters: re-encode one subsample at two
budgets and compare the same clips to themselves. Not worth a GPU here, because
of the logical point below.

Second caveat, on power: audio-only R@1 of 0.25% is ~130 hits across the whole
set, so bucket-level differences are noisy. The direction is consistent across
four buckets and is not the sharp degradation the hypothesis requires.

And the logical point that should have come first: FLARE's protocol is the
official codebase at default configuration. **If the paper also ran defaults, the
defaults cancel** and cannot produce a gap between us at all. A shared default is
not a discrepancy.

So the audio deficit stays open. Of the three attractive explanations, one is
ruled out by inspection (no shared cap exists in the adapter), one by measurement
(wav spans match their clips exactly), and one is ruled out *as an explanation of
the gap* by logic rather than by data — a shared default cannot produce a
discrepancy with a paper that used the same defaults, whatever the constants
are.

That is what an honest reproduction section looks like: the deficit is real,
audio-correlated, biases our numbers conservatively, and is unexplained.

## The WAVE contrast: measured

Both single-modality arms are complete at 399/399 videos, 53,580 queries, on the
released evaluation code at the 64-frame protocol default.

| arm | published | ours | Δ | Δ relative |
|---|---|---|---|---|
| vision-only R@1 | 16.80 | **16.5995** | −0.200 | −1.19% |
| vision-only R@5 | 37.54 | 36.8664 | −0.674 | −1.80% |
| vision-only R@10 | 47.92 | 47.3591 | −0.561 | −1.17% |
| audio-only R@1 | 7.58 | **7.4057** | −0.174 | −2.30% |
| audio-only R@5 | 18.41 | 17.8163 | −0.594 | −3.23% |
| audio-only R@10 | 24.80 | 23.9978 | −0.802 | −3.23% |

The harness gate passes on the corrected rule: no same-sign truncation
signature beyond what the deficit already carries, and the audio arm landed at
7.4057 inside a pre-registered 7.39–7.45.

**The deficit is real at depth and is not numeric.** This is worth separating
from the rank-1 story, because at R@1 alone it looks explainable and is not.
Measured on this exact data, changing *only* the scoring precision from fp32 to
bf16 — identical embeddings — moves R@1 by −0.105 (audio) and −0.095 (vision),
and reorders the top-1 result for 10.1% and 8.7% of queries. So both R@1 gaps
sit at roughly two units of that floor. But the same perturbation costs
**nothing** at deeper cutoffs: ΔR@5 of +0.040 / −0.040 and ΔR@10 of +0.050 /
+0.055, i.e. sign-unstable noise. The observed R@5 and R@10 deficits are −0.59
to −0.80, ten to twenty times the floor.

A numeric account therefore explains R@1 and explains nothing below it. Writing
"reproduces within numeric reproducibility" would have been wrong, and it is
what I had written before checking every cutoff.

**What was eliminated, by measurement rather than argument.** The frame/segment
budget (a 32-vs-64 control on the same gallery: +0.015 at R@1); clip boundary
alignment (extracted audio and extracted video agree to 11 ms median, 30 ms
worst, across 250 clips, all 16 kHz mono); tie structure (0.000% of queries have
rank-1 and rank-2 exactly equal — FLARE is not MultiVENT); the attention rewrite
(bitwise identical to the stock masked path for single-segment inputs, which is
every audio clip); the query prompt and `max_pixels` (both upstream, introduced
in the initial code release, so common-mode with the anchor); and the BEATs
tower (the snapshot ships zero `beats.*` tensors, so the bundled checkpoint is
the sole and correct source).

Checkpoint identity, which the reproduction section previously could not state:
`tsinghua-ee/WAVE-7B` at revision `7d51cdaecfaabb9c529a447249cd4c2a6df8ce5b`.

**One deviation remains and it is a hardware consequence, not a protocol
choice.** The adapter hardcodes `attn_implementation="sdpa"`, upstream — so the
published numbers did *not* come from WAVE's flash-attention training default.
Stock masked SDPA hands a `[1, n, n]` mask to the kernel, which forces the math
backend and materialises `num_heads · n²` scores: a single 47 GiB allocation on
a 44 GiB card. Attending per `cu_seqlens` segment with no mask is the same
arithmetic in 19 GB. This is consistent with the authors running 80 GB cards,
where the allocation simply fits.

What is left for the deep-rank deficit contains nothing we chose: locally
extracted media, weights relative to whatever the paper actually ran (no
revision is published), or non-attention numerics.

**Fusion, provisional.** The tuned scalar sweep over WAVE's own vision-only and
audio-only embeddings peaks near `w ≈ 0.65` at R@1 ≈ 20.9. That clears the
pre-registered "below 29.7" decisively. It is provisional in two ways: it is the
linear family only, and on ImageBind the renormalized family beat linear at the
optimum and flipped a probe's sign, so treat 20.9 as a lower bound; and the
comparison should be made against our own joint row, not the published 42.63,
since cross-configuration levels carry the floor above while same-configuration
contrasts over shared embeddings do not.

Report the gain, not the level. Both routes start from vision-only at 16.60, and
fusion does not get credit for a baseline it never had to earn:

| route | R@1 | buys over best single channel |
|---|---|---|
| tuned late fusion | ~20.9 | +4.3 |
| native joint encoding | 42.63 | +26.0 |

**Late fusion recovers about 16.5% of the joint-encoding gain**, not the ~49%
that 20.9/42.63 suggests. A linear operator over frozen unimodal embeddings
captures roughly a sixth of what joint encoding achieves on a substrate where
both channels are alive — which is the operator-route-versus-encoder-route
question answered against the linear operator, and answered on published anchors
at both ends.

## Which models collapse, and why: a published 2×2

The same table answers a question the note had been treating as open. Table 4,
T→Clip, query regime, R@1:

| model | fused | vision-only | fusion does |
|---|---|---|---|
| ImageBind | 6.35 | 12.87 | **hurts 2.03×** |
| LanguageBind | 3.32 | 15.52 | **hurts 4.67×** |
| Perception AV Large | 7.79 | 7.22 | helps 1.08× |
| Wave-7B | 42.63 | 16.80 | helps 2.54× |

The split is not "which encoders have live audio". Appendix B.1 states the
mechanism outright: ImageBind and LanguageBind ship no official recipe for
combining per-modality embeddings, so **the benchmark's authors** average-pooled
the ℓ₂-normalized vision and audio vectors themselves, following standard
late-fusion practice. Perception AV Large and Wave-7B natively emit a joint
audiovisual embedding and are used as-is.

**Every model where fusion was improvised collapses; every model with a native
joint encoder gains.** 2 for 2 on each side.

This sharpens the claim and fixes a scope risk at the same time. The honest
statement is *not* "audiovisual fusion is hard" or even "hand-designed operators
lose". It is: **the collapse is a property of the mean-pooling stopgap applied to
encoders that have no joint output.** Nothing here licenses a conclusion about
audiovisual fusion in general — the two models that actually fuse jointly both
improve, one of them by 2.5×.

Which is also why the operator question moves to WAVE rather than staying on
ImageBind: a scalar repairs the stopgap, and the interesting question is whether
score-level combination of two live channels can reach what joint training
reaches.

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


## Controls re-derived on the renormalized family

Recomputed after the family error, so ceiling, null and grouping arms all come
from the operator the benchmark actually uses. The headroom moved from +4.83 to
**+4.64** — both ends shifted.

| | R@1 |
|---|---|
| best fixed `w` ≈ 0.87–0.90 | 13.12 |
| per-query oracle **within grid** | 17.76 |
| **decoy null, same grid, same family** | **0.01** |
| lexical audio proxy, 2 groups | 13.11 (−0.01) |
| `max q·v`, 5 bins | 13.10 (−0.01) |
| `max q·a`, 5 bins | 13.06 (−0.06) |
| audio affinity, 2 / 5 bins | 13.09 / 13.10 (−0.03 / −0.02) |

**Which family wins is now measured, not asserted:** renormalized 13.12 against
linear 12.86, paired **+0.256, 95% CI [+0.121, +0.396]**, excluding zero.

Why the oracle here is a *grid* quantity and never an "exact" one: the exact
per-query solver used on the linear family required the score to be affine in
`w`, so each distractor gave a half-line and the feasible set was a single
interval. Renormalized, the denominator is per-document and does not cancel in
`s_g > s_d` — the feasible set need not be an interval or even connected.
Reusing that solver would return a confident wrong ceiling shaped exactly like a
right one, which is the retracted MultiVENT "+1.69 hard ceiling" failure again:
a quantity that measured the sampler rather than the bound.

Five groupings, none beating a single global weight, against a null-controlled
+4.64 of available headroom. The conclusion survives the family correction.


## The flip survives five fold partitions

A reversal carrying a claim deserves more than one split. The paired bootstrap
resamples *queries*; it never resamples the *split*. And the seed test was
already a no-op once here — deterministic full-batch Adam, sd exactly 0.00 — so
the fold partition is the only randomness actually exercising this estimator.
Five different video→fold assignments, refit end to end:

| partition | const | `w(q)` | `w(q)` − const | vs best FIXED w=0.87 |
|---|---|---|---|---|
| 0 | 13.09 | 13.23 | +0.149 [+0.050, +0.246] | +0.116 [+0.017, +0.218] |
| 1 | 13.07 | 13.23 | +0.164 [+0.067, +0.267] | +0.116 [+0.017, +0.216] |
| 2 | 13.08 | 13.23 | +0.149 [+0.054, +0.244] | +0.114 [+0.013, +0.215] |
| 3 | 13.08 | 13.23 | +0.146 [+0.048, +0.239] | +0.110 [+0.011, +0.213] |
| 4 | 13.06 | 13.25 | +0.185 [+0.088, +0.280] | +0.131 [+0.030, +0.230] |

**All five positive, every CI excluding zero, mean +0.159 (sd 0.015).** It also
beats the best *fixed* grid point (13.12), not just the held-out constant, by
+0.110 to +0.131 — so it is not winning against a handicapped baseline.

Two things this corrects or qualifies in my own favour and against it:

**The first partition was at the optimistic end.** I reported +0.192 from a
single split; the five-partition mean is **+0.159**, and that is the number to
quote. Small, but it is the fourth time on this benchmark that a value from one
configuration read higher than the honest aggregate.

**"Query-conditioned" should not be oversold.** The head emits w with sd 0.039
around a mean of 0.887 — full range 0.571 to 0.977, so it is genuinely varying
and not a constant with jitter, but for most queries it is modulating by about
±0.04. A real effect of a modest size, produced by a modest deviation from the
best constant.
