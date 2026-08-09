# 06 — Long-video frame selection: the rule inverts with evidence structure

**Verdict: the learned operator does not beat the hand-designed ones, the
selection effect over random subsets is about 2 points, and the question
"which hand-designed one" has no fixed answer.** The best selection
rule flips depending on how many pieces of evidence a question needs, which is
why the whole field's gains cluster at 1–2 points. One effect here is
significant, and it is not about selection at all.

## Setup

Frozen `Qwen2.5-VL-7B-Instruct`, scored by **letter logits** (one forward,
argmax over the option letters present — no generation, no regex).
`SigLIP-so400m` features for every method, budget 16 frames, so the only thing
varying between rows is which frames get picked. LongVideoBench val, long
split (600 s + 3600 s, n=976). Training-free rows are deterministic; paired
McNemar throughout. **Trained rows are single-seed — by rule 4 of AGENTS.md
they are not results, and are marked as such.**

Harness check first: uniform sampling at budgets {8,16,32,64} reproduces
published numbers for this model (58.3 @64 vs 56.0–59.2 reported).

## 1. Candidate density (a control, not a finding — this is prior art)

Same rule, only the candidate pool changes:

| rule | pool 64 | pool 512 (1 fps) |
|---|---|---|
| uniform-16 | 51.7 | 51.4 |
| SigLIP top-16 | 53.0 | **57.7** |
| AKS | 53.7 | 56.8 |
| BOLT | 54.0 | 55.5 |

Top-*k* gains +4.7 overall, +6.0 at one hour, χ²=11.0, purely from a denser
shortlist. Uniform is flat, as it must be.

**This is not ours.** Frame-Voyager swept the candidate pool 8→256 on
Video-MME (47.5→50.8) and reports saturation past 128; ReQuest ablates fps and
frames it as robustness. ReQuest further shows more frames actively *hurting*
past 1024. We report it as a control establishing that our pool is not the
binding constraint, and because the size of the effect relative to every
algorithmic difference is a caution for anyone comparing selectors at a sparse
pool — not as a contribution.

## 2. The ranking inverts with evidence count (the main finding)

Prior work has the *phenomenon*: Adaptive Greedy Frame Selection
(arXiv:2603.20180) reports that different MLVU task categories prefer
relevance-heavy or coverage-heavy presets, and ships a question-type router.
What follows is the *mechanism* — evidence count is the latent variable behind
that heterogeneity, and unlike a task label it is causal, measurable per item,
and not tied to one dataset's taxonomy. Their own categories switch preference
between backbones; evidence count should not.

LVB annotates each question's evidence timestamps. Stratifying by how many:

| evidence positions | n | uniform | top-*k* | AKS | BOLT | AdaRD-key | cDPP |
|---|---|---|---|---|---|---|---|
| single (1) | 439 | 57.9 | **69.9** | 65.6 | 63.3 | 61.7 | 69.2 |
| two (2) | 377 | 49.9 | 54.1 | **54.4** | 53.8 | 54.1 | 54.1 |
| multi (≥3) | 160 | 37.5 | **32.5** worst | 38.1 | 38.1 | **39.4** best | 35.6 |

Relevance top-*k* is best by 8 points on single-evidence questions and **worst,
below uniform,** on multi-evidence ones. Same reversal in two independent cuts
(annotated reasoning level; temporal span of the evidence). Within the ≥3
stratum alone the pairwise test is ns (n=160); the pattern is credible because
three independent cuts agree, not from that cell.

45% of long items are single-needle, so aggregates are dominated by the regime
where concentrating is correct.

The number with no analogue in prior work is that relevance top-*k* falls
**below uniform** on multi-evidence questions. A per-category router can pick
the better of two rules; it cannot reveal that the field's default rule is
worse than no rule at all on 16% of the data.

Headroom: best single method 57.9 · oracle routing over question type 58.9 ·
**per-item oracle over these six methods 74.0**. A 12-line keyword router
switching between two existing methods scores 58.1, beating every single one.

## 3. A cheap index of the *unsampled* video (significant)

Measured token prices: 1 frame ≈ 360 tok, 1 subtitle span ≈ 8 tok, so a whole
hour-long transcript ≈ 1250 tok ≈ 3.5 of the 16 frames. Adding it:

| evidence positions | Δ |
|---|---|
| single (1) | −2.7 ns — a distraction |
| two (2) | −1.6 ns |
| **multi (≥3)** | **+10.0, χ²=5.92 significant** |

Inside that stratum the gain is **not** on subtitle-grounded questions
(+3.7 ns) but on **visually** grounded ones (+13.2, χ²=9.39). So the transcript
is not answering anything — it is a cheap temporal *index* of what happened
between the 16 frames the model can see. Same inversion as §2, different
dimension (which stream to buy), and this one is significant.

## What did not work

All single-seed; treat as directional.

| Attempt | Outcome |
|---|---|
| REINFORCE fine-tune of a competitive init (K=2, no whitening, no trust region) | **−6.0**; 75% of the damage by step 400; in-domain holdout never showed it |
| + advantage whitening, KL trust region, entropy target | Stops the damage, produces no gain — policy barely moves |
| Conditioning the mixture on pool geometry (length, density) | **Hurts**: 53.1 vs 54.2 unconditioned |
| Conditioning the mixture on the question | Only method that is top on single-evidence (70.2) *and* beats top-*k* on multi (36.9 vs 32.5) — but regresses on 2-position items, loses the aggregate, all ns. A 12-line temporal-NMS heuristic matches it |
| Off-policy replay instead of on-policy sampling | Same accuracy, **11× cheaper** (0.9 s/step, zero LM forwards while training), moves the learned potentials far more. If you do this, use replay |

## What was not controlled

One frozen answerer; one benchmark; one encoder. Trained rows are single-seed.
Evidence positions are benchmark annotations, not verified minimal sufficient
sets. The ≥3 stratum is 160 items. Thumbnail and learned-aggregation variants
were still running when this was written.

## Reusable conclusions

1. **Report frame-selection results stratified by evidence count.** An
   aggregate averages two regimes with opposite optima and will show ~1 point
   for almost any method. This is our best guess at why the field's numbers
   cluster.
2. **Check candidate density before crediting an algorithm.** Density moved
   more than any algorithmic difference we measured, and it was the only
   significant selection-side effect.
3. **When your initialization is already competitive, an unregularized
   policy-gradient fine-tune can only lose.** Measure drift-from-init and
   target-distribution accuracy from step one; an in-domain holdout will not
   warn you.
4. **The learned × set-level cell is empty in this literature.** Every trained
   selector we found scores frames independently and adds diversity afterwards
   by a fixed rule (NMS, stratified top-*k*, segment allocation); every method
   that genuinely models the set is hand-designed. We tried to fill that cell
   and tied.
5. **Budget in tokens, not frames.** Streams have wildly different prices, and
   the cheapest one bought the only significant gain we found.

## Prior work that overlaps this note

- **Adaptive Greedy Frame Selection** (arXiv:2603.20180): submodular
  relevance+coverage over a 1 fps pool with a text-only question-type router
  over four fixed (α,β) presets. Has the heterogeneity phenomenon by MLVU
  category. No evidence-count stratification, no pool-density study, no code.
- **ReQuest** (arXiv:2607.01737, ECCV 2026): question-adaptive NMS where the
  spacing is set by the *answer entropy* of a first uniform pass. Adapts a
  tradeoff, but on a scalar difficulty signal, with independent per-frame
  scoring and post-hoc NMS. Their rule concentrates when entropy is high; if
  multi-evidence questions are the high-entropy ones, §2 predicts it misfires
  exactly there. We have not tested that yet and flag it as a prediction, not
  a result.
- **Frame-Voyager** (arXiv:2410.03226): the candidate-pool sweep in §1.
- **GenEvA** (arXiv:2607.28516): query-conditioned latent slots appended to a
  frozen Video-MLLM. Aggregates the *selected* frames; the open question here
  is the unsampled remainder.

## Note on other people's methods

We reimplemented AKS, BOLT, a conditional DPP and AdaRD-key under one protocol;
deviations from their papers (encoder, pool size, scoring) are in the code, so
these are our numbers under our conditions, not reproductions of theirs. Two
observations, offered as reproducibility notes: trained selectors in this
literature are generally compared against uniform and CLIP top-*k* rather than
against the stronger training-free methods, which sit ~1.5 points above uniform
in our hands; and where selection-only gain is isolated against a *frozen*
answerer it is small (~+1 point in one paper's own ablation, with the larger
headline requiring the answerer to be fine-tuned as well).
