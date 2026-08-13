# Publication candidates

When a line finds something worth publishing, add it here by PR, in this format:
**what we found** (one sentence a non-specialist can read), **why it matters**
(one sentence), the numbers, the note, and citations for any claim about what
the field does. The critic verifies the controls; Idan decides the venue.
Assembly drafts live in `papers/`.

---

## Ready — numbers on main, instruments validated

### 1. "Oracle headroom" can be an illusion

**What we found:** for 9 of 10 MultiVENT queries, SOME weighting of the four
channel scores would rank the right video first (89.3%, vs 0.8% for a random
decoy) — but a model trained to produce that weighting from the query alone
gains nothing at ANY capacity we could make learn: the full ladder — linear /
MLP-256 / token-attention heads on pre-norm features, 3 seeds, video-disjoint
5-fold — tops out at 56.74 ± 0.00 (pre-norm linear) vs plain max 57.43, with
capacity only adding variance (MLP-256: 54.00 ± 3.88; attention: 55.16 ± 1.33)
and the post-norm linear anchor reproducing (56.25 ± 0.02 vs the original
56.23). The headroom is real geometry and absent information — un-decodable
from the query across the ladder.

**Why it matters:** oracle and upper-bound gaps are standard motivation in
selection/fusion work — Frame-Voyager builds its training signal from
oracle-style ranking of frame combinations
([2410.03226](https://arxiv.org/abs/2410.03226)), and our own ledger fell for
one twice (note 04's retracted +1.69 "hard ceiling"; note 06's retracted
74.0 per-item oracle, 98% chance harvesting). We show such gaps can be
unreachable in principle from the query side, and we ship the one-eval test
(LP feasibility + decoy null + learnability probe). (Note 04.)

### 2. Frame selection is worth ~2.5 points, not the ~10 it appears

**What we found:** on 4-option MCQ, two equal-quality runs already disagree on
~8% of items by chance (churn floor 8.12 on Qwen2.5-VL-7B); the raw flip-set
reads 10.7 points but the churn-corrected market for choosing WHICH 16 frames
that frozen model sees is 2.5 ± 1 on LongVideoBench
([2407.15754](https://arxiv.org/abs/2407.15754)). The market is
CHECKPOINT-DEPENDENT: on Qwen3-VL-8B the churn-corrected market rises to
**3.8 ± 1** (raw b16→b64 net 5.33) — Qwen2.5-VL was saturated, not the task.
Qwen3-VL's floor is now measured and the answer is the opposite of what we
expected: **8.44 vs 8.12**, so churn is a property of the 4-option MCQ format
rather than of the model, and floors DO transfer even though markets do not.
Caveat kept: on Qwen3-VL the `lost` term (6.86) sits with 55 of 56 floor pairs
above it, outside the floor's bulk where Qwen2.5-VL's sat centred — the same
disjoint→nested transfer slack seen in the budget-32 replication — so quote
2.5 → 3.8, both ±1, not two precise numbers.

**Why it matters:** the frame-selection literature's typical claimed deltas are
1–2 points against uniform or CLIP top-k — AKS
([2502.21271](https://arxiv.org/abs/2502.21271)), Frame-Voyager
([2410.03226](https://arxiv.org/abs/2410.03226)), Q-Frame
([2506.22139](https://arxiv.org/abs/2506.22139)), QCA
([2607.00983](https://arxiv.org/abs/2607.00983)), FOCUS
([2510.27280](https://arxiv.org/abs/2510.27280)), AGFS
([2603.20180](https://arxiv.org/abs/2603.20180)), ReQuest
([2607.01737](https://arxiv.org/abs/2607.01737)) — i.e., inside or near the
noise floor we measure on the anchor where the floor exists — AND the market
itself is checkpoint-dependent, so single-generation selection results do not
transfer. Two axes, one audit. None of the cited papers report a paired null
of this kind. Precedent
for the audit-paper shape in another domain: "A Sober Look at Progress in
Language Model Reasoning" ([2504.07086](https://arxiv.org/abs/2504.07086)).
(Note 06.)

### 3. The field's default frame-picking rule inverts on multi-evidence questions

**What we found:** query–frame similarity top-k (the default rule, as in AKS
[2502.21271](https://arxiv.org/abs/2502.21271) and Q-Frame
[2506.22139](https://arxiv.org/abs/2506.22139)) beats uniform spacing by 8
points on questions needing ONE moment and loses to coverage-aware rules on
questions needing ≥3 — and benchmark averages cancel the two regimes.

⛔ **The strong form is retired, as underpowered rather than as disproved.**
We previously claimed top-k falls BELOW uniform on multi-evidence. That cell is
n=160 and needs n≈17,818 for within-cell significance, so neither the original
−5.0 (Qwen2.5-VL: 32.5 vs 37.5) nor the Qwen3-VL reversal (+1.8: 36.2 vs 34.4)
is distinguishable from noise. The honest statement is **no support on either
generation once powered correctly** — and that is itself the strata argument,
since the cell was quotable precisely because nobody powered it.

**The survivor is the ranking reversal**, which has real amplitude and holds on
both generations: the top-k − AdaRD-key gap swings +8.2 → −6.9 between the
single- and multi-evidence strata on Qwen2.5-VL, and +4.3 → −6.3 on Qwen3-VL.
Relevance-heavy rules win localized evidence and lose distributed evidence;
coverage-heavy rules do the reverse.

**Why it matters:** leaderboard averages cannot distinguish a better method
from a luckier mix of question types; the stratifying variable (evidence count)
is annotated in LongVideoBench and computable elsewhere. AGFS
([2603.20180](https://arxiv.org/abs/2603.20180)) observed per-category
preference heterogeneity; evidence count is the per-item causal variable
behind it. (Note 06.)

### 4. Which long-video results survive a model generation — and which are leaderboard noise

**What we found:** we re-ran the same protocol across two model generations with
the frame selections held byte-identical (they are SigLIP-derived, so
model-independent — only the answering model changes). Long split, n=976,
budget 16.

| result | Qwen2-VL-7B (2024) | Qwen2.5-VL-7B (2025) | Qwen3-VL-8B (Oct 2025) | verdict |
|---|---|---|---|---|
| candidate density buys | +6.15 | +6.0 | +5.8 | **survives, flat** |
| churn floor | 8.64 | 8.12 | 8.44 | **survives, flat** (format property, not model) |
| ranking reversal, top-k − AdaRD across strata | — | +8.2 → −6.9 | +4.3 → −6.3 | **survives** |
| selection market, *disjoint-pairs floor* | +2.02 | +2.54 | +3.75 | grows |
| selection market, *nested (`lost`) floor* | +3.18 | +2.77 | +5.33 | grows, not monotone |
| timestamped transcript, multi-evidence | — | +10.0 (χ²=5.92) | +2.5 (ns) | flips |

⚠️ **Two floor conventions, shown deliberately, because they disagree.** The
corrected market is `recover − floor`. Taking the floor from disjoint random
same-budget pairs gives the monotone series; taking it as the nested `lost` term
gives `recover − lost`, the raw net, which is not monotone. They diverge exactly
where the transfer slack is largest — `lost` sits at the edge of its floor on 2
of 3 models (51/56 and 55/56 pairs above it; only Qwen2.5-VL is centred at
42/56), which is an argument for the nested reading.

**What is robust to both conventions, and all we claim:** Qwen3-VL's market
exceeds *both* earlier generations under either floor, while density does not
move. Strict monotonicity rests on the 2.02-vs-2.54 gap and is one floor-choice
away from reversing, so we do not assert it.

**And the ordering of the methods themselves is not resolvable.** Within
Qwen2.5-VL, of all pairs among {NMS-2s, cDPP, top-16, AKS, BOLT, AdaRD-key}
exactly one clears paired McNemar — NMS-2s over AdaRD-key, +3.18, χ²=4.76.
Within Qwen3-VL **none clears**: the whole six-method table, spanning 56.8 to
59.4, is one statistical blob. And the single separable pair on the older model
is dead even on the newer one (−0.10, χ²=0.00). Rank changes people would
report as findings — cDPP overtaking NMS, AdaRD-key rising from last to third —
are shuffles among differences of 0.1–0.4 with χ² < 0.1.

**Pre-registered for the fourth anchor** (Qwen3.6-35B-A3B, 2026-04), stated on
quantities that do not inherit the market's ±1 error: (a) market(Qwen3.6) >
market(Qwen3-VL) means the growth continues, ≤ means it flattens; (b) density
stays in [5.5, 6.5]; (c) churn floor stays in [8, 9]. We deliberately do *not*
pre-register a density:market ratio — with ±1 on the market, 6.15/(2.02±1) spans
2.0× to 6.0×, so a ratio trend would be illustration rather than measurement.

**Why it matters:** this is the two-axis audit. On the strata axis, benchmarks
cannot resolve published method deltas *within* strata (entry 3's power table:
1–3 point method differences need 500–18,000 items per stratum; LongVideoBench
supplies 160 and 439). On the generation axis, the results that do reach
significance divide cleanly into structural ones that transfer and
capacity-limited ones that do not. A reader can use the split to decide which
of their own results to distrust: anything whose size is set by what the model
*cannot yet do* is a statement about a checkpoint. The suggestive form — the
market's upper envelope grows across generations while density stays flat, so
selection may be closing on sampling — is offered as a hypothesis for the fourth
anchor to test, not as a measured trend.
(Note 06, `papers/swing3-strata-spec.md`.)

---

## Pending — expected to reach this bar

### FLARE: audio-visual fusion collapses are a missing hyperparameter — and the per-query fix is unlearnable

> ⚠️ **FAMILY ERROR, CORRECTED — two items still open.** Numbers with a `w`
> attached were originally computed on the LINEAR family
> `w·(q·v) + (1−w)·(q·a)`, while the incumbent is `q · normalize(w·v + (1−w)·a)`.
> The two coincide only at w=0 and w=1, so both endpoints reproduced and every
> calibration check passed while the entire middle of the curve was a different
> operator. Re-derived on the incumbent's own family: best fixed operator
> **≈13.1 (plateau w ≈ 0.87–0.90)**, not the ~12.8 linear plateau; audio's
> nested selection-free margin **+0.474, 95% CI [+0.271, +0.664]**, roughly
> double the linear family's +0.254; grid oracle **17.76**, headroom +4.64.
> Drop 12.81, 12.86 and ~12.8 wherever they appear — all were optima of a
> family the benchmark does not use. STILL OPEN: the probe null (12.49) and the
> eight groupings are linear-family and pair with a bar that no longer exists,
> and the decoy null has not been re-run on this family, so the 17.76 oracle
> currently has no matched null. Treat the unlearnability claim as provisional
> until both land.


**What we found:** on the first benchmark whose queries require BOTH vision and
audio (FLARE, [2605.10228](https://arxiv.org/abs/2605.10228)), the standard
fusion — average of L2-normalized embeddings, the recipe FLARE itself calls
"standard late-fusion practice" — HALVES ImageBind's performance on a fixed
query set (R@1 12.61 vision-only → 5.86 fused, 2.15×; 3.0× on vision-targeted
queries). One scalar repairs it: down-weighting audio recovers ~12.8 on a
PLATEAU, w ∈ [0.90, 0.95] (101-point grid, 5-fold video-disjoint, tuned on
train folds and scored held out). We do not quote a single argmax: paired
bootstrap on w=0.93 vs w=0.95 gives +0.049, CI [−0.028, +0.123], which includes
zero — our own earlier "12.81 → 12.86 at w=0.93" correction was false precision
of exactly the kind it was correcting. What survives the same test is audio's
marginal contribution over vision alone, measured under NESTED selection —
`w` re-chosen off-fold and re-chosen again inside every bootstrap resample, so
the tuning step carries its own variance: **+0.254, 95% CI [+0.063, +0.386]**,
excluding zero. Fixed-weight robustness lines beneath it: 0.90/0.93/0.95 give
+0.194/+0.254/+0.205. Significant but tiny — and not the 6.75-point hole the
collapse appears to leave.

Then the ceiling. A per-query oracle that picks the best weight for each query
separately reaches 18.50, a genuine +5.64 over the best constant, computed
EXACTLY rather than sampled (with two modalities the score is affine in w, so
each distractor is a half-line and the feasible set is a single interval). It
is null-controlled: the same procedure aimed at random non-gold targets
harvests 0.01, and only 0.01% of queries have an exact duplicate of the gold.
But it is not reachable. A linear head on the query embedding, initialized AT
the best constant so that doing nothing was a free pass, instead drifted
(‖θ‖=7.9, w spread 0.63–0.96) and generalized BELOW its own initialization:
12.49 held out, recovering −0.35 of the +5.64.

**Why it matters:** two claims, both actionable. First, the incumbent is not
near-optimal (ViDoRe-style) but BADLY CHOSEN — deployments using mean fusion
pay the measured 2.15–3.0× for a missing hyperparameter with a one-line fix.
Second, FLARE closes in MultiVENT's shape — oracle-reachable,
probe-unrecoverable — making it our second data point for the general lesson
that an oracle ceiling is not a target unless something visible at inference
time predicts it. One open tension before that line can stand as stated: our
own free-ride result shows fusion moves vision-targeted and audio-targeted
queries in OPPOSITE directions, and query type is not hidden at inference, so a
2–3 parameter per-group weight is the obvious thing the linear probe should
have found and did not. That test is minutes and is pending; if a per-group
weight recovers a meaningful share of the +5.64, "probe-unrecoverable" narrows
to "this probe, this parameterization". The audit (fixed-query/varied-media screen + per-channel
operator rows + dead-channel instrument outcomes, note 05) ports to any AV
retrieval stack.

**Mechanism — one measured fact, one open split.** Measured and solid: the same
averaging that halves vision-targeted queries RESCUES audio-targeted ones 8×
(0.12 → 0.97). That free-ride asymmetry is the load-bearing evidence, and it
rests on which queries move, not on any score algebra.

The score algebra we previously offered was WRONG and is retracted. We wrote
that normalized-mean score = (q·v + q·a)/√(2+2·v·a) with measured
cos(v,a)=0.25 gives "0.63× margin compression", predicting
12.61 × 0.6325 = 7.98 against 5.86 observed. R@1 is invariant to any positive
rescaling of all scores for a query, and computing 0.63 from the MEAN cos(v,a)
treats the divisor as a global constant — under exactly that assumption its
effect on R@1 is ZERO, not −37%. The prediction and the assumption used to
derive it are incompatible, so there is no 7.98 anchor and no 0.735 residual to
explain. Compression is real for margins and irrelevant for ranks.

Rebuilt, only two things can move R@1 here: (a) the injected q·a term, and
(b) HETEROGENEITY of v_i·a_i across gallery items. **The control is now RUN**
(same 53,580 queries × 87,697 clips; it took 8 CPU-minutes, not the GPU pass we
budgeted for):

| condition | score | R@1 |
|---|---|---|
| vision only | q·v | 12.61 |
| dead audio, compression kept | q·v / ‖v+a‖ | **10.65** |
| incumbent | (q·v + q·a) / ‖v+a‖ | 5.86 |
| audio added, no compression | (q·v + q·a) / 2 | 5.00 |

Harness check: the formula reproduces the arm's own measured incumbent to +0.00
against its encoded gallery, so this describes the real operator rather than an
algebraic lookalike.

**Injected noise is the major term and compression is the minor one.** Of the
6.75-point collapse, the normalizer costs 1.96 (29%) and adding q·a costs 4.79
(71%). Our retracted algebra had predicted 7.98 for the dead-audio row; the
measured value is 10.65, so that residual was not merely unmeasured, it pointed
the wrong way.

And the split is NON-ADDITIVE, with an order-dependent sign. Read the other
way round: adding audio to an UN-normalized score costs 7.61 (12.61 → 5.00),
and then switching to the per-document normalizer GIVES BACK 0.86 (5.00 →
5.86). On an already audio-polluted score, the varying denominator HELPS. So
the "documents are penalised for disagreeing with themselves" story is a small
effect whose sign depends on the order you apply it in, and it is not the
account of this collapse. What survives is blunter and is what the paper should
say: **the operator's failure is that it adds a noise channel at equal weight.**
That is why a scalar repairs it, and why the repair is a WEIGHT rather than a
better normalizer. Measured cos(v,a) = 0.247 ± 0.120.

**The per-group weight is now run too, and it is also null — which is what
promotes "probe-unrecoverable" from a probe artifact to a claim.** The obvious
objection to a failed linear probe is that query TYPE is visible at inference
and moves the two groups oppositely, so a 2–3 parameter per-group weight should
walk away with the headroom. It does not. Beyond a lexical proxy (+0.01), seven
NON-SEMANTIC groupings — binning queries by their own retrieval statistics with
no lexicon: max q·v (2/5/10 bins), max q·a (5 bins), audio affinity q·a − q·v
(2/5/10 bins) — score +0.00, −0.04, −0.12, −0.11, −0.03, −0.05, −0.04 against a
single global w, versus +4.83 available on the grid. None beats one number,
most are worse, and FINER BINNING DEGRADES held-out performance: a per-group
argmax fits its bin and does not transfer. So the ceiling resists both a
learned per-query head and hand-specified per-group structure, on the axis
where our own free-ride result says the signal must live.

**Corrections we carry rather than bury.** Our "12.81 at w=0.95, found
independently twice" was a SHARED BLIND SPOT: two sweeps agreed because neither
grid contained 0.93, so the agreement was one measurement run twice, not a
cross-check. Treat identical numbers from a shared pipeline as a reproduction
until the inputs are shown disjoint. The FIX for that was then wrong in the
same family: "12.86 at w=0.93" quoted an argmax whose lead over 0.95 is
+0.049, CI [−0.028, +0.123], i.e. indistinguishable from zero. A denser grid
buys resolution, not significance; the honest object is the plateau, and the
test that separates them is a paired bootstrap, which is what promoted audio's
marginal contribution from claim to result. That fix needed two further passes.
Quoting the argmax's +0.25 still inherits the selection that produced 0.93, and
a pre-specified weight only trades one selection for a weaker one — 0.95 was
itself the argmax of a coarser grid over the same queries. Nested selection
settles it by measurement: **+0.254, CI [+0.063, +0.386]**. The instructive part
is where the bias actually lived. The point estimate never moved — all five
folds pick the same `w`, so off-fold selection returns the plug-in answer — but
holding that `w` fixed while resampling made the interval **1.13× too narrow**,
putting the lower bound at +0.112 where the honest one is +0.063. We spent three
passes hunting an overstated effect; what was overstated was the confidence.

**WAVE-7B status, and a log that outran three readings of it.** The contrast is
RUNNING as of 2026-08-13 12:48 — four sharded gallery workers, results
climbing from zero — after an SDPA fix plus 4-way sharding. Every earlier
statement in this ledger about WAVE described a run that no longer exists, and
the tell was in the filename: the error log we all quoted is
`errors_preSDPAfix.txt`, i.e. explicitly marked superseded. Three readings of
it went wrong in three different ways, worth recording because they are
independent failure modes: (i) we first reported "all three arms OOM" from
sampling ONE arm's log, when the true split is maudio 1596 dtype/0 OOM,
munified 399 dtype/0 OOM, mvision 399 OOM/0 dtype — two arms were a missing
fp16 cast in `WAVEAdapter._move_inputs`, not memory; (ii) we then quoted
"allocations up to 22.34 GiB" from four sampled lines, when the actual
distribution over 399 OOM lines is min 2.60 / median 18.46 / MAX 47.27 GiB,
with 45 attempts requesting MORE THAN THE ENTIRE 44.39 GiB CARD — those can
never be fixed by freeing memory and needed a structural split, which is what
sharding supplied; and (iii) all of it was read as current when the file name
said otherwise. A stale artifact answers every question you ask it, in the
present tense.

**Retracted novelty claim.** This entry previously said the controlled
vision-vs-fused contrast was new and that the paper never ran it query-based.
False: FLARE's Table 4 carries both sides for every V+A model, and the collapse
was visible in the published appendix the whole time. Read the appendix before
claiming novelty.

What that leaves is a better harness check and a narrower contribution. Our arms
reproduce BOTH published sides: fused 5.86 vs 6.35 (−7.7%), vision-only 12.61 vs
12.87 (−2.0%) — two-sided agreement inside 8%. Note both deviations are
NEGATIVE, so they read as a small systematic offset rather than noise; worth
locating, and worth remembering when the WAVE arms are gated against their own
published anchors.

The contribution is therefore not the observation but the explanation, the
repair, and the bound on repairing further — and the 4-pager should lead with
the last of those, since it is the only one a reviewer cannot find in the
benchmark's own appendix: the per-query optimum is worth a further +4.64 on the
grid, and neither a learned head nor eight visible groupings recover any of it —
subject to the two re-derivations named in the warning above. Venue: ICASSP 2027 (submission 2026-09-16), with
the operator question scoped to live-channel substrates (WAVE, once its dtype
cast and its OOM are both fixed).
Caveats: linear probe, single frozen query embedding, single seed, score-level
only — token-level and candidate-aware operators are untested and remain the
live option.

### Imaginability audit (H3 line, design locked)

Swap a benchmark's real video for one GENERATED from the question alone
(MiniMax H3); if the score barely drops, the benchmark is not testing video
understanding. A format-matched, portable validity meter for any video
benchmark — constructive where TVBench
([2410.07752](https://arxiv.org/abs/2410.07752)) diagnosed MVBench
([2311.17005](https://arxiv.org/abs/2311.17005)) destructively (shuffle,
single-frame, blind). Adjacent trend it audits rather than joins: video models
as reasoners — Veo 3 ([2509.20328](https://arxiv.org/abs/2509.20328)), Thinking
with Video ([2511.04570](https://arxiv.org/abs/2511.04570)).

### "Search can't say no" (NegBench line, substrate gate first)

Every indexable retrieval scorer — bi-encoder cosine (CLIP,
[2103.00020](https://arxiv.org/abs/2103.00020)), late-interaction MaxSim
(ColBERT, [2004.12832](https://arxiv.org/abs/2004.12832)), sparse expansion
(SPLADE, [2107.05720](https://arxiv.org/abs/2107.05720)) — is monotone:
matching more never lowers a score, so "a photo without a frisbee" scores
HIGHER on frisbee photos. NevIR ([2305.07614](https://arxiv.org/abs/2305.07614))
and NegBench ([2501.09425](https://arxiv.org/abs/2501.09425)) measure the
failure (near-chance); the 2026 SOTA fixes are non-indexable listwise LLM
rerankers (SIGIR'25 reproduction,
[2502.13506](https://arxiv.org/abs/2502.13506)) — "saying no costs quadratic".
We formalize the impossibility and test the cheapest indexable signed fix on
frozen encoders (NevIR frozen prior: +3.7).
