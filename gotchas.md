# Gotchas that produce *plausible wrong numbers*

The dangerous bugs are not the ones that crash. These all ran to completion and
returned numbers a reasonable person would believe.

## Silently randomly-initialized weights

`transformers` 4.52 renamed Qwen2.5-VL's submodule path
(`model.layers.*` → `model.language_model.layers.*`). Any **subclass** that
bypasses the rename mapping loads **zero** pretrained weights, prints a warning,
and carries on.

**Narrowed by a second data point (thanks to the frame-selection line): the
hazard is SUBCLASSING, not the version.** Using
`Qwen2_5_VLForConditionalGeneration` directly on transformers 4.57.6 is fine —
they reproduce published LongVideoBench numbers (58.3 @64 frames), which is
impossible with random weights. So do **not** pin 4.51.3 reflexively; pin only if
you are loading through a subclass that reimplements `forward` (CLaMR's
`ColQwen2_5` does). The general check below is what matters, not the version.

Result was nDCG@10 0.2244 instead of 0.5847 — low, but not obviously broken,
because identical tokens still get identical *random* embeddings, so
late-interaction degenerates into random-projection lexical matching.

**Check:** grep model-loading output for `newly initialized`, and compare
checkpoint keys to `model.state_dict()` keys before believing any number.

## Left padding + last-token pooling

VLM2Vec sets `padding_side="left"` and pools "the largest index where
`mask == 1`". Using `mask.sum(1) - 1` (the token *count*) is correct **only for
the longest sequence in the batch**; everything else pools a pad position.

**Symptom:** cosine to reference embeddings is ~0.99 for one row in the batch and
0.3–0.5 for the rest.

## Visual inputs dropped by a tensor filter

Some processors return `pixel_values` as a **list of numpy arrays**, not a
tensor. A filter like `{k: v for k, v in inputs.items() if torch.is_tensor(v)}`
silently deletes the images, and every candidate encodes to the same text-only
vector.

**Symptom:** pairwise cosine among embeddings of *different* images ≈ 0.999
(reference: ~0.35). Always sanity-check that different inputs give different
embeddings.

## Any max-over-alternatives quantity needs a null at the same cardinality

Found independently by two lines, in two different shapes. Report the quantity
**minus its null**, never the raw quantity.

**Oracles.** On multiple choice the null harvests the guessing rate: a subset
oracle over 6 *random* draws reached 73.49 where 6 *real* selection methods
reached 73.76 — a claimed ~16 points of headroom was almost entirely lottery.

**Flip sets.** Same trap, different shape, and easy to miss because no oracle is
involved. "Items that budget B gets wrong and budget 64 gets right" looked like
10 points of headroom on LongVideoBench — but two equal-strength 16-frame runs
already churn **8.12** points by guessing alone, leaving ~2.5. Measure the churn
floor with independent draws at the same budget.

**Retrieval.** The guessing floor is usually too small to harvest (top-10 of 1504
= 0.66%), but the null still earns its keep by exposing error **correlation**:
four real MultiVENT channels scored **12.7 points below** four noise-matched
noisy copies of a single channel. Redundancy isn't "they overlap" — it's "they
share a failure mode".

Check the floor's *distribution*, not its mean. On LongVideoBench, across all 56
ordered pairs of independent 16-frame draws the floor ranges 6.76–9.84 (sd 0.63),
and against it `recover` = 10.66 is z = +4.0 with **0 of 56** pairs reaching it,
while `lost` = 7.89 is z = −0.4 with 42 of 56 above it. That separation is the
evidence.

(An earlier version of this entry cited two "independent" checks — `lost` landing
on the floor at −0.23, and the corrected ceiling 54.18 closing on uniform@64
54.41. Those are the *same* check: since acc@64 = acc@16 + recover − lost,
corrected_ceiling − acc@64 IS lost − floor, so both print −0.23 by construction.
Worth knowing as its own trap: an identity will happily masquerade as
corroboration.)

## A no-op edit that still commits

`str.replace` on a missing anchor changes nothing and raises nothing. If the same
commit touches another file, it succeeds and looks like the fix landed. This is
how a retraction of a *false published claim* silently failed here for two
commits — the anchor had been rewritten by a merged PR in between.

Assert the anchor before replacing, and re-grep the file after pushing. The
failure is invisible in `git log`, which is exactly why it survives.

## Single-seed results

Seed sd was ~1 point on MMEB VisDial reranking. A first-seed "+2.3 points"
became **+0.27, p=0.22** at n=16. See [03](03-mmeb-visdial.md).

Prefer **paired** designs — same seed, same data order, one thing varied — which
make the difference far less noisy than two independent runs.

## Convex pooling cannot beat a max

`softmax(logits) · row ≤ max(row)`, always. If your learned reduction is a convex
combination and your baseline is `max`, the experiment is unwinnable by
construction and a flat sweep tells you nothing about the world. Use a zero-gated
**unbounded** residual: `score = max(...) + g · Δ`, `g = 0` at init.

## Zero-init deadlock

Zero-initializing both the gate and the residual's output layer gives both zero
gradient forever. Zero exactly one side.

A second form: a zero-init *projection* does not deadlock, it silently starves
everything upstream. Our aggregation head showed the projection taking 100% of
the gradient share and the attention machinery feeding it exactly 0% — it would
have trained the projection over a frozen random head and produced a plausible
mediocre number. Check gradient *shares*, not gradient existence.

## Contrastive loss saturating on cosines

Cosine similarities in [-1, 1] make cross-entropy saturate at ≈ ln(N) with no
gradient. Add a learnable logit scale (CLIP-style, init ~20).

## Storage

`hf_hub_download(cache_dir=...)` returns a symlink into a blob store; unlinking
the returned path frees nothing. Use `local_dir=<tempdir>` and `rmtree` it if you
are streaming hundreds of shards through a shared volume.

## qwen-vl-utils video min_pixels floor

Independently hit by two projects, so worth the exact numbers. In
qwen-vl-utils 0.0.14, `VIDEO_MIN_TOKEN_NUM = 128`, so the per-frame floor is
`128 × 28 × 28 = 100,352` px. If you pass only `max_pixels` (e.g. `224*224 =
50,176` for small frames, or `3136` for 4-token thumbnails), `max < min` and
`smart_resize` raises `AssertionError: The max_pixels of image must be greater
than or equal to min_pixels` on **every** item.

The error text does not say which minimum it means. Fix: set **both** in the same
element dict — `{"min_pixels": 784, "max_pixels": 3136}` yields a real ~3-token
frame. And note there are two call sites in CLaMR (`process_combined` and
`process_frames`); patching one is not enough.

## Trainer side effects

HF `Trainer` auto-enables `wandb` when it is installed, and can fail at the
*logging* callback after a complete, expensive eval pass. `--report_to none`.

## Masking with `-inf` turns a KL term into NaN once batches are ragged

`p · (log p − log q)` with padding masked as `-inf` evaluates
`exp(-inf) · (-inf − finite)` = `0 · -inf` = NaN, which then poisons every
downstream gate. Invisible with fixed-size inputs, because nothing is ever
padded; fires on almost every step once elements have different lengths. Mask
with a large **finite** value and zero the padded terms before summing. Add a
regression test on a deliberately ragged batch — a smoke test at fixed size
cannot see this class of bug.

## Feature caches keyed on the wrong id

Check `len(unique ids) == len(items)` before trusting one. A popular
video-instruction set keys its annotations on *video* id, not item id: 31,008
questions collapsed onto 7,774 feature files, later questions silently
overwriting earlier ones, and a reward cache keyed on that id mixed rewards
across different questions of the same video. Training then runs happily on
inputs conditioned by the wrong question.

## A selection experiment must change *which* items are fed, never *how*

Feeding chosen video frames as a pre-decoded image list double-resizes them
relative to the library's own file-reading path: 94% prediction agreement with
the baseline instead of ~99% — a few points of apparent "method effect" that is
really preprocessing. Patch the reader to honour explicit indices *inside* the
normal path, and verify agreement on a slice before running anything.

## A subset oracle on multiple choice harvests chance

`oracle@R = max over R candidate subsets of per-item correctness` gives every
wrong item R independent ~25% lottery tickets, and the inflation grows with R —
it is far larger than the 25% floor and subtracting an analytic chance term does
not remove it. The control is a **null oracle**: max over R *random* subsets at
the same R. Report `oracle − null_oracle`.

We claimed ~16 points of selection headroom this way. With the control, six
random 16-frame subsets scored 73.49 against six real methods' 73.76 — the
headroom was ~98% lottery, and the honest effect was the R=1 gap of ~2 points.
The failure is silent: the inflated number looks like strong motivation for the
whole research direction.

## A per-query linear oracle is not a ceiling for per-item non-linear reductions

The complement of the entry above: that one inflates, this one under-bounds.
An oracle over per-query channel weights (note 04's 59.12) bounds only scoring
functions *linear* in the channel scores with per-query coefficients. `min` and
`product` are per-item non-linear — which channel binds varies per item within
one query's pool — and can legitimately exceed such a "ceiling". A per-item
oracle is no fix (it degenerately ranks golds first); evaluate fixed non-linear
operators directly.

The geometry that decides it, in two channels: gold G = (0.6, 0.6) against
distractors D1 = (0.8, 0.45), D2 = (0.45, 0.8). G beats D1 under weights
(a, 1-a) only for a < 3/7, and beats D2 only for a > 4/7 — no weighting ranks
G first; `min` does (0.6 vs 0.45). The binding condition is distractors
**Pareto-flanking** the gold, not "the query is conjunctive": against the
textbook AND-failure distractors (1, 0) and (0, 1), any a in (0.4, 0.6) works
and linear wins fine. Two of our own first constructions fell exactly this way.

So screen for it operator-free: per query, count golds with no feasible linear
weighting (an LP with one constraint per distractor — works at any channel
count) that a monotone non-linear reduction still ranks first. Material count:
weighted means are structurally out and a min-family or learned reduction has
real room. Count ~0: tune a scalar and stop. Calibrate the diagnostic on
MultiVENT's dumped tensor first, where redundancy (note 04) predicts ~0.
